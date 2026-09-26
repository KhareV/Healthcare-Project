"""Persistent longitudinal health-record storage (MongoDB), optional.

Layering matters here: this module stores only *raw* structured input a
user entered (profile fields, conditions, encounter observations/support
intervals, report metadata) and *prediction-run snapshots* -- never the
frozen scientific pipeline's derived representation (canonical feature
rows, SHAP matrices, model internals). When a persisted encounter needs to
be served again (typically after an API restart, since V2ServingRuntime's
serving state is in-memory only), the exact same registration path used at
creation time (serving.v2.custom_record) regenerates that derived
representation from the stored raw input. MongoDB is never read by the
scientific pipeline itself and no frozen artifact is touched by this
module.

Conceptual shape (see docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md):

    Clerk user (owner_user_id)
      -> one patient_profiles document (lazily created on first use)
           -> many conditions
           -> many encounters (each with embedded observations/support_intervals)
                -> many prediction_runs
           -> many reports (metadata only; binary content lives in
              serving.v2.report_storage, never in Mongo)
      -> audit_events (a lightweight trail of meaningful mutations)

Disabled gracefully: if MONGODB_URI is unset, or the cluster is unreachable
at startup, `enabled` is False and every method either no-ops or raises
PersistenceUnavailableError -- callers must treat that as "durability
unavailable" and degrade accordingly, matching the project's existing
pattern for optional external services (Groq, Kokoro, Clerk-local-dev
fallback).

Every document is scoped by `owner_user_id`, the same verified-Clerk-token
identity used everywhere else in this product -- this module never accepts
an owner id from a request body, only from its caller (api/v2_app.py, which
derives it from ClerkAuthenticator). NEVER log or persist a bearer token, a
Clerk secret, or a MongoDB connection string anywhere in this module.
"""

from __future__ import annotations

import functools
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Callable, List, Mapping, Optional, Sequence, TypeVar

logger = logging.getLogger("serving.v2.persistence")

DB_NAME = "prt_v2_health_records"

_F = TypeVar("_F", bound=Callable)


_RETRY_BACKOFF_SECONDS = (0.25, 0.75, 1.5)


def _retrying(fn: _F) -> _F:
    """Retries a Mongo operation up to three times, with increasing pauses,
    on a transient connection error (e.g. `AutoReconnect: connection pool
    paused`, observed in practice when several requests hit an Atlas
    connection concurrently right after a brief network blip or topology
    change -- not hypothetical, reproduced under the seven-way concurrent
    read My Health Record's overview issues on every page load). This turns
    a rare transient failure into a slightly slower successful response
    instead of a hard 500 -- this API layer previously had no resilience to
    this at all, surfacing it directly to the caller."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from pymongo.errors import AutoReconnect, ConnectionFailure

        last_exc: Exception = None  # type: ignore[assignment]
        for attempt, pause in enumerate((0.0, *_RETRY_BACKOFF_SECONDS)):
            if pause:
                time.sleep(pause)
            try:
                return fn(*args, **kwargs)
            except (AutoReconnect, ConnectionFailure) as exc:
                last_exc = exc
                logger.warning("transient MongoDB connection error in %s (attempt %d): %s", fn.__name__, attempt + 1, type(exc).__name__)
        raise last_exc

    return wrapper  # type: ignore[return-value]

PATIENT_PROFILE_SCHEMA_VERSION = "patient_profile_v1"
CONDITION_SCHEMA_VERSION = "condition_v2"
ENCOUNTER_SCHEMA_VERSION = "encounter_v2"
REPORT_SCHEMA_VERSION = "report_v1"

CONDITION_STATUSES = ("active", "resolved", "historical")


class PersistenceUnavailableError(RuntimeError):
    """Raised when a write/read is attempted while persistence is disabled."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _without_mongo_id(doc: Mapping[str, object]) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8).upper()}"


class MongoPersistence:
    """A thin, owner-scoped wrapper over a MongoDB Atlas cluster."""

    def __init__(self, uri: Optional[str]):
        self.enabled = False
        self._client = None
        if not uri:
            return
        from pymongo import ASCENDING, MongoClient
        from pymongo.errors import PyMongoError

        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            db = client[DB_NAME]
            db.patient_profiles.create_index([("owner_user_id", ASCENDING), ("patient_id", ASCENDING)], unique=True)
            db.conditions.create_index([("owner_user_id", ASCENDING), ("patient_id", ASCENDING)])
            db.conditions.create_index([("owner_user_id", ASCENDING), ("status", ASCENDING)])
            db.encounters.create_index([("owner_user_id", ASCENDING), ("stay_id", ASCENDING)], unique=True)
            db.encounters.create_index([("owner_user_id", ASCENDING), ("patient_id", ASCENDING)])
            db.prediction_runs.create_index(
                [("owner_user_id", ASCENDING), ("stay_id", ASCENDING), ("prediction_time", ASCENDING)], unique=True
            )
            db.reports.create_index([("owner_user_id", ASCENDING), ("report_id", ASCENDING)], unique=True)
            db.reports.create_index([("owner_user_id", ASCENDING), ("patient_id", ASCENDING)])
            db.reports.create_index([("owner_user_id", ASCENDING), ("report_date", ASCENDING)])
            db.audit_events.create_index([("owner_user_id", ASCENDING), ("timestamp", ASCENDING)])
        except PyMongoError as exc:
            logger.error("MongoDB unavailable at startup; persistence disabled (%s)", type(exc).__name__)
            return
        self._client = client
        self.enabled = True

    @property
    def _db(self):
        return self._client[DB_NAME]

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise PersistenceUnavailableError("MongoDB persistence is not configured or unreachable")

    # -- patient profile ---------------------------------------------------
    # One profile per Clerk user for this pass (deliberately not a
    # multi-patient system yet -- see docs/product_v2/
    # HEALTH_RECORD_ARCHITECTURE.md -- but the unique index is already
    # (owner_user_id, patient_id), so a future multi-profile mode needs no
    # index changes). Demographics use age_years (consistent with every
    # other demographic field already in this synthetic-benchmark product,
    # e.g. demo patients and custom records) rather than a date_of_birth,
    # since a calendar birth date is never otherwise meaningful in this
    # pipeline and would imply a real-world precision this synthetic system
    # does not have.

    @_retrying
    def get_profile(self, *, owner_user_id: str) -> Optional[Mapping[str, object]]:
        doc = self._db.patient_profiles.find_one({"owner_user_id": owner_user_id})
        return _without_mongo_id(doc) if doc else None

    @_retrying
    def get_or_create_profile(self, *, owner_user_id: str) -> Mapping[str, object]:
        existing = self.get_profile(owner_user_id=owner_user_id)
        if existing is not None:
            return existing
        now = _utcnow_iso()
        doc = {
            "patient_id": _new_id("PAT"),
            "owner_user_id": owner_user_id,
            "display_name_or_alias": "My Health Record",
            "age_years": None,
            "sex_category": None,
            "blood_group": None,
            "height_cm": None,
            "weight_kg": None,
            "created_at": now,
            "updated_at": now,
            "schema_version": PATIENT_PROFILE_SCHEMA_VERSION,
        }
        self._db.patient_profiles.insert_one(dict(doc))
        return doc

    @_retrying
    def update_profile(self, *, owner_user_id: str, fields: Mapping[str, object]) -> Mapping[str, object]:
        profile = self.get_or_create_profile(owner_user_id=owner_user_id)
        allowed = {"display_name_or_alias", "age_years", "sex_category", "blood_group", "height_cm", "weight_kg"}
        update = {k: v for k, v in fields.items() if k in allowed}
        update["updated_at"] = _utcnow_iso()
        self._db.patient_profiles.update_one(
            {"owner_user_id": owner_user_id, "patient_id": profile["patient_id"]}, {"$set": update}
        )
        return self.get_profile(owner_user_id=owner_user_id)

    # -- conditions ----------------------------------------------------

    @_retrying
    def add_condition(
        self, *, owner_user_id: str, patient_id: str, name: str, code: Optional[str],
        diagnosed_date: Optional[str], status: str, notes: Optional[str],
    ) -> Mapping[str, object]:
        now = _utcnow_iso()
        doc = {
            "condition_id": _new_id("COND"),
            "owner_user_id": owner_user_id,
            "patient_id": patient_id,
            "name": name,
            "code": code,
            "diagnosed_date": diagnosed_date,
            "status": status,
            "notes": notes,
            "created_at": now,
            "updated_at": now,
            "schema_version": CONDITION_SCHEMA_VERSION,
        }
        self._db.conditions.insert_one(dict(doc))
        return _without_mongo_id(doc)

    @staticmethod
    def _normalize_condition(doc: Mapping[str, object]) -> dict:
        """Migration-on-read for conditions created under the earlier
        (label/diagnosed_year, Mongo-_id-addressed) shape -- never mutates
        the stored document, only the value returned to callers."""

        legacy_id = str(doc["_id"]) if "condition_id" not in doc and "_id" in doc else None
        row = _without_mongo_id(doc)
        if legacy_id is not None:
            row["condition_id"] = legacy_id
        if "name" not in row:
            row["name"] = row.get("label", "")
        row.setdefault("code", None)
        if "diagnosed_date" not in row and row.get("diagnosed_year"):
            row["diagnosed_date"] = f"{row['diagnosed_year']}-01-01"
        row.setdefault("diagnosed_date", None)
        row.setdefault("notes", None)
        row.setdefault("patient_id", None)
        row.setdefault("updated_at", row.get("created_at"))
        row.setdefault("schema_version", "condition_v1")
        return row

    @staticmethod
    def _condition_filter(owner_user_id: str, condition_id: str) -> dict:
        """Matches either the current condition_id field or, for a
        condition created before that field existed, its Mongo _id given as
        a hex string -- every condition in a database created before this
        pass predates condition_id, so this fallback is load-bearing, not
        theoretical."""

        from bson import ObjectId
        from bson.errors import InvalidId

        clauses: List[dict] = [{"condition_id": condition_id}]
        try:
            clauses.append({"_id": ObjectId(condition_id)})
        except InvalidId:
            pass
        return {"owner_user_id": owner_user_id, "$or": clauses}

    @_retrying
    def list_conditions(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        docs = list(self._db.conditions.find({"owner_user_id": owner_user_id}).sort("created_at", 1))
        return [self._normalize_condition(d) for d in docs]

    @_retrying
    def get_condition(self, *, owner_user_id: str, condition_id: str) -> Optional[Mapping[str, object]]:
        doc = self._db.conditions.find_one(self._condition_filter(owner_user_id, condition_id))
        return self._normalize_condition(doc) if doc else None

    @_retrying
    def update_condition(self, *, owner_user_id: str, condition_id: str, fields: Mapping[str, object]) -> Optional[Mapping[str, object]]:
        allowed = {"name", "code", "diagnosed_date", "status", "notes"}
        update = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not update:
            return self.get_condition(owner_user_id=owner_user_id, condition_id=condition_id)
        update["updated_at"] = _utcnow_iso()
        result = self._db.conditions.update_one(self._condition_filter(owner_user_id, condition_id), {"$set": update})
        if result.matched_count == 0:
            return None
        return self.get_condition(owner_user_id=owner_user_id, condition_id=condition_id)

    @_retrying
    def delete_condition(self, *, owner_user_id: str, condition_id: str) -> bool:
        result = self._db.conditions.delete_one(self._condition_filter(owner_user_id, condition_id))
        return result.deleted_count > 0

    # -- encounters (one document per custom record) ---------------------

    @_retrying
    def save_encounter(
        self, *, owner_user_id: str, patient_id: str, stay_id: str, subject_id: str, patient_alias: str,
        age_years: int, sex_category: str, intime: str, outtime: str,
        encounter_type: str = "MANUAL_RECOVERY_EPISODE", source: str = "MANUAL_STRUCTURED_ENTRY",
        observations: Sequence[Mapping[str, object]] = (), support_intervals: Sequence[Mapping[str, object]] = (),
    ) -> None:
        now = _utcnow_iso()
        self._db.encounters.update_one(
            {"owner_user_id": owner_user_id, "stay_id": stay_id},
            {
                "$set": {
                    "patient_id": patient_id, "subject_id": subject_id, "patient_alias": patient_alias,
                    "age_years": age_years, "sex_category": sex_category,
                    "intime": intime, "outtime": outtime,
                    "encounter_type": encounter_type, "cardiac_condition_group": "CUSTOM_RECORD", "source": source,
                    "observations": list(observations), "support_intervals": list(support_intervals),
                    "updated_at": now, "schema_version": ENCOUNTER_SCHEMA_VERSION,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    @staticmethod
    def _normalize_encounter(doc: Mapping[str, object]) -> dict:
        row = _without_mongo_id(doc)
        row.setdefault("patient_id", None)
        row.setdefault("encounter_id", row.get("stay_id"))
        row.setdefault("encounter_type", "MANUAL_RECOVERY_EPISODE")
        row.setdefault("source", "MANUAL_STRUCTURED_ENTRY")
        row.setdefault("updated_at", row.get("created_at"))
        row.setdefault("schema_version", "encounter_v1")
        row.setdefault("observations", [])
        row.setdefault("support_intervals", [])
        return row

    @_retrying
    def get_encounter(self, *, stay_id: str) -> Optional[Mapping[str, object]]:
        """Looked up by stay_id alone (not owner) -- rehydration must find
        the encounter's *true* owner before any ownership check can happen;
        the caller (serving.v2.custom_record.rehydrate_if_needed) restores
        it into the runtime under that true owner, and the existing
        api/v2_app.py::_authorize_custom_stay comparison against the
        caller's verified token is what actually enforces access -- this
        method itself makes no authorization decision."""

        doc = self._db.encounters.find_one({"stay_id": stay_id})
        return self._normalize_encounter(doc) if doc else None

    @_retrying
    def list_encounters(self, *, owner_user_id: str, patient_id: Optional[str] = None) -> List[Mapping[str, object]]:
        """Migration-on-read: an encounter persisted before patient_id
        existed is transparently backfilled to the caller's profile (there
        is exactly one profile per owner in this pass, so this is
        unambiguous) the first time it is listed -- a real, additive write,
        never a destructive rewrite of anything else on the document."""

        query: dict = {"owner_user_id": owner_user_id}
        docs = list(self._db.encounters.find(query).sort("created_at", 1))
        needs_backfill = [d for d in docs if not d.get("patient_id")]
        if needs_backfill:
            profile = self.get_or_create_profile(owner_user_id=owner_user_id)
            for d in needs_backfill:
                self._db.encounters.update_one(
                    {"_id": d["_id"]}, {"$set": {"patient_id": profile["patient_id"], "updated_at": _utcnow_iso()}}
                )
                d["patient_id"] = profile["patient_id"]
        rows = [self._normalize_encounter(d) for d in docs]
        if patient_id is not None:
            rows = [r for r in rows if r.get("patient_id") == patient_id]
        return rows

    # -- observations & support intervals (derived views over encounters) --

    @_retrying
    def list_observations(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        """A flattened, chronological view across every encounter's
        embedded observations -- observations are not a separate
        collection (they stay embedded in their owning encounter, matching
        the existing Mongo schema), this is purely a read-side projection."""

        rows: List[dict] = []
        for encounter in self.list_encounters(owner_user_id=owner_user_id):
            for obs in encounter.get("observations", []):
                rows.append({
                    "encounter_id": encounter.get("encounter_id"),
                    "patient_alias": encounter.get("patient_alias"),
                    "concept": obs.get("concept"),
                    "value": obs.get("value"),
                    "hours_since_admission": obs.get("hours_since_admission"),
                })
        rows.sort(key=lambda r: (r["encounter_id"] or "", r["hours_since_admission"] or 0))
        return rows

    @_retrying
    def list_support_intervals(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        rows: List[dict] = []
        for encounter in self.list_encounters(owner_user_id=owner_user_id):
            for interval in encounter.get("support_intervals", []):
                rows.append({
                    "encounter_id": encounter.get("encounter_id"),
                    "patient_alias": encounter.get("patient_alias"),
                    "kind": interval.get("kind"),
                    "agent": interval.get("agent"),
                    "rate": interval.get("rate"),
                    "start_hour": interval.get("start_hour"),
                    "end_hour": interval.get("end_hour"),
                })
        rows.sort(key=lambda r: (r["encounter_id"] or "", r["start_hour"] or 0))
        return rows

    # -- prediction-run snapshots ------------------------------------------

    @_retrying
    def save_prediction_run(self, *, owner_user_id: str, stay_id: str, prediction_time: str, snapshot: Mapping[str, object]) -> None:
        self._db.prediction_runs.update_one(
            {"owner_user_id": owner_user_id, "stay_id": stay_id, "prediction_time": prediction_time},
            {"$set": {**snapshot, "updated_at": _utcnow_iso()}, "$setOnInsert": {"created_at": _utcnow_iso()}},
            upsert=True,
        )

    @_retrying
    def list_prediction_runs(self, *, owner_user_id: str, stay_id: Optional[str] = None) -> List[Mapping[str, object]]:
        query: dict = {"owner_user_id": owner_user_id}
        if stay_id is not None:
            query["stay_id"] = stay_id
        docs = list(self._db.prediction_runs.find(query).sort("prediction_time", 1))
        return [_without_mongo_id(d) for d in docs]

    # -- reports (metadata only; binary content is in report_storage) ------

    @_retrying
    def add_report(
        self, *, owner_user_id: str, patient_id: str, title: str, document_type: str,
        original_filename: str, mime_type: str, storage_key: str, size_bytes: int, report_date: Optional[str],
    ) -> Mapping[str, object]:
        now = _utcnow_iso()
        doc = {
            "report_id": _new_id("RPT"),
            "owner_user_id": owner_user_id,
            "patient_id": patient_id,
            "title": title,
            "document_type": document_type,
            "original_filename": original_filename,
            "mime_type": mime_type,
            "storage_key": storage_key,
            "size_bytes": size_bytes,
            "report_date": report_date,
            "uploaded_at": now,
            "source": "MANUAL_UPLOAD",
            "processing_status": "NOT_PARSED",
            "extracted_summary": None,
            "schema_version": REPORT_SCHEMA_VERSION,
        }
        self._db.reports.insert_one(dict(doc))
        return _public_report(doc)

    @_retrying
    def list_reports(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        docs = list(self._db.reports.find({"owner_user_id": owner_user_id}).sort("uploaded_at", -1))
        return [_public_report(d) for d in docs]

    @_retrying
    def get_report(self, *, owner_user_id: str, report_id: str) -> Optional[Mapping[str, object]]:
        doc = self._db.reports.find_one({"owner_user_id": owner_user_id, "report_id": report_id})
        return _without_mongo_id(doc) if doc else None

    @_retrying
    def delete_report(self, *, owner_user_id: str, report_id: str) -> Optional[Mapping[str, object]]:
        """Returns the deleted document's full metadata (including
        storage_key, needed by the caller to also delete the underlying
        file) if a matching, owned report existed; None otherwise."""

        doc = self._db.reports.find_one_and_delete({"owner_user_id": owner_user_id, "report_id": report_id})
        return _without_mongo_id(doc) if doc else None

    # -- report intelligence: candidate measurements (never auto-inserted) --
    # A candidate is stored on its report's own document (no new collection
    # needed) as `candidate_measurements`, each with a `confirmed: bool`.
    # Only `confirm_report_candidates` ever flips that flag, and only after
    # the API layer has copied the confirmed values into an encounter's real
    # observations (serving.v2.custom_record.append_observations_to_encounter)
    # -- this method itself never touches an encounter.

    @_retrying
    def save_report_candidates(self, *, owner_user_id: str, report_id: str, candidates: Sequence[Mapping[str, object]], status: str) -> None:
        self._db.reports.update_one(
            {"owner_user_id": owner_user_id, "report_id": report_id},
            {"$set": {"candidate_measurements": list(candidates), "processing_status": status, "parsed_at": _utcnow_iso()}},
        )

    @_retrying
    def confirm_report_candidates(self, *, owner_user_id: str, report_id: str, candidate_ids: Sequence[str]) -> None:
        if not candidate_ids:
            return
        self._db.reports.update_one(
            {"owner_user_id": owner_user_id, "report_id": report_id},
            {"$set": {"candidate_measurements.$[elem].confirmed": True}},
            array_filters=[{"elem.candidate_id": {"$in": list(candidate_ids)}}],
        )

    # -- encounters: appending confirmed report measurements ----------------

    @_retrying
    def append_encounter_observations(self, *, owner_user_id: str, stay_id: str, observations: Sequence[Mapping[str, object]]) -> None:
        """Extends an existing encounter's raw observations (e.g. from
        confirmed report candidates) -- additive only, never replaces or
        reorders what was already there."""

        self._db.encounters.update_one(
            {"owner_user_id": owner_user_id, "stay_id": stay_id},
            {"$push": {"observations": {"$each": list(observations)}}, "$set": {"updated_at": _utcnow_iso()}},
        )

    # -- audit events --------------------------------------------------

    def record_event(self, *, owner_user_id: str, patient_id: Optional[str], action: str, target_type: str, target_id: str) -> None:
        """Best-effort: a failure here must never break the mutation it is
        recording. Never pass raw report content, tokens, or secrets as
        `target_id`/`action` -- only identifiers and a short action label."""

        try:
            self._db.audit_events.insert_one({
                "event_id": _new_id("AUD"),
                "owner_user_id": owner_user_id,
                "patient_id": patient_id,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "timestamp": _utcnow_iso(),
            })
        except Exception:  # noqa: BLE001
            logger.warning("failed to record audit event: %s %s", action, target_type)

    @_retrying
    def list_audit_events(self, *, owner_user_id: str, limit: int = 200) -> List[Mapping[str, object]]:
        docs = list(self._db.audit_events.find({"owner_user_id": owner_user_id}).sort("timestamp", -1).limit(limit))
        return [_without_mongo_id(d) for d in docs]

    # -- export --------------------------------------------------------

    @_retrying
    def export_record(self, *, owner_user_id: str) -> Mapping[str, object]:
        """Everything Part 16 requires, structured JSON only -- report
        binaries are never embedded, only their metadata."""

        profile = self.get_or_create_profile(owner_user_id=owner_user_id)
        encounters = self.list_encounters(owner_user_id=owner_user_id)
        predictions: List[dict] = []
        for encounter in encounters:
            predictions.extend(self.list_prediction_runs(owner_user_id=owner_user_id, stay_id=encounter["stay_id"]))
        return {
            "schema_version": "health_record_export_v1",
            "exported_at": _utcnow_iso(),
            "profile": profile,
            "conditions": self.list_conditions(owner_user_id=owner_user_id),
            "encounters": encounters,
            "reports": self.list_reports(owner_user_id=owner_user_id),
            "prediction_history": predictions,
        }


def _public_report(doc: Mapping[str, object]) -> dict:
    """Report metadata safe to return to a client -- never the internal
    storage_key (a logical key, not a real filesystem path, but still not
    something a client needs or should be able to use directly; downloads
    always go through the authorized /download endpoint)."""

    row = _without_mongo_id(doc)
    row.pop("storage_key", None)
    return row
