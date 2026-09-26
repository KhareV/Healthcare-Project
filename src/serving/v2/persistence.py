"""Optional persistent longitudinal storage for custom records (MongoDB).

Layering matters here: this module stores only the *raw* structured input a
user entered (observations, support intervals, profile fields, conditions)
and *prediction-run snapshots* -- never the frozen scientific pipeline's
derived representation (canonical feature rows, SHAP matrices, model
internals). When a persisted encounter needs to be served again (typically
after an API restart, since V2ServingRuntime's serving state is in-memory
only), the exact same registration path used at creation time
(serving.v2.custom_record) regenerates that derived representation from the
stored raw input. MongoDB is never read by the scientific pipeline itself
and no frozen artifact is touched by this module.

Disabled gracefully: if MONGODB_URI is unset, or the cluster is unreachable
at startup, `enabled` is False and every method is a safe no-op / returns
an empty result -- callers must treat that as "durability unavailable" and
fall back to the original in-memory-only behavior, matching the project's
existing graceful-degradation pattern for optional external services
(Groq, Kokoro, Clerk-local-dev fallback). A record is never lost by this
layer being down: creation and serving still work, they just do not
survive a restart.

Every document is scoped by `owner_user_id`, the same verified-Clerk-token
identity used everywhere else in this product -- this module never accepts
an owner id from a request body, only from its caller (api/v2_app.py, which
derives it from ClerkAuthenticator).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Mapping, Optional, Sequence

logger = logging.getLogger("serving.v2.persistence")

DB_NAME = "prt_v2_health_records"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _without_mongo_id(doc: Mapping[str, object]) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


class MongoPersistence:
    """A thin, owner-scoped wrapper over a MongoDB Atlas cluster.

    Three collections: `conditions` (a user's persistent profile-level
    diagnosis list), `encounters` (one document per custom record, holding
    the raw observations/support-intervals the user entered), and
    `prediction_runs` (one snapshot per distinct cutoff a custom encounter
    was actually replayed at, upserted so re-visiting a cutoff never
    duplicates history).
    """

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
            db.conditions.create_index([("owner_user_id", ASCENDING)])
            db.encounters.create_index([("owner_user_id", ASCENDING), ("stay_id", ASCENDING)], unique=True)
            db.prediction_runs.create_index(
                [("owner_user_id", ASCENDING), ("stay_id", ASCENDING), ("prediction_time", ASCENDING)], unique=True
            )
        except PyMongoError as exc:
            logger.error("MongoDB unavailable at startup; persistence disabled (%s)", type(exc).__name__)
            return
        self._client = client
        self.enabled = True

    @property
    def _db(self):
        return self._client[DB_NAME]

    # -- conditions (profile-level, not tied to any one encounter) -------

    def add_condition(self, *, owner_user_id: str, label: str, diagnosed_year: Optional[int], status: str) -> Mapping[str, object]:
        from bson import ObjectId

        oid = ObjectId()
        doc = {
            "_id": oid, "owner_user_id": owner_user_id, "label": label,
            "diagnosed_year": diagnosed_year, "status": status, "created_at": _utcnow_iso(),
        }
        self._db.conditions.insert_one(dict(doc))
        result = _without_mongo_id(doc)
        result["condition_id"] = str(oid)
        return result

    def list_conditions(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        docs = list(self._db.conditions.find({"owner_user_id": owner_user_id}).sort("created_at", 1))
        out = []
        for doc in docs:
            row = _without_mongo_id(doc)
            row["condition_id"] = str(doc["_id"])
            out.append(row)
        return out

    def delete_condition(self, *, owner_user_id: str, condition_id: str) -> bool:
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            oid = ObjectId(condition_id)
        except InvalidId:
            return False
        result = self._db.conditions.delete_one({"_id": oid, "owner_user_id": owner_user_id})
        return result.deleted_count > 0

    # -- encounters (one document per custom record) ---------------------

    def save_encounter(
        self, *, owner_user_id: str, stay_id: str, subject_id: str, patient_alias: str,
        age_years: int, sex_category: str, intime: str, outtime: str,
        observations: Sequence[Mapping[str, object]], support_intervals: Sequence[Mapping[str, object]],
    ) -> None:
        now = _utcnow_iso()
        self._db.encounters.update_one(
            {"owner_user_id": owner_user_id, "stay_id": stay_id},
            {
                "$set": {
                    "subject_id": subject_id, "patient_alias": patient_alias,
                    "age_years": age_years, "sex_category": sex_category,
                    "intime": intime, "outtime": outtime,
                    "observations": list(observations), "support_intervals": list(support_intervals),
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def get_encounter(self, *, stay_id: str) -> Optional[Mapping[str, object]]:
        """Looked up by stay_id alone (not owner) -- rehydration must find
        the encounter's *true* owner before any ownership check can happen;
        the caller (serving.v2.custom_record.rehydrate_if_needed) restores
        it into the runtime under that true owner, and the existing
        api/v2_app.py::_authorize_custom_stay comparison against the
        caller's verified token is what actually enforces access -- this
        method itself makes no authorization decision."""

        doc = self._db.encounters.find_one({"stay_id": stay_id})
        return _without_mongo_id(doc) if doc else None

    def list_encounters(self, *, owner_user_id: str) -> List[Mapping[str, object]]:
        docs = list(self._db.encounters.find({"owner_user_id": owner_user_id}).sort("created_at", 1))
        return [_without_mongo_id(d) for d in docs]

    # -- prediction-run snapshots ------------------------------------------

    def save_prediction_run(self, *, owner_user_id: str, stay_id: str, prediction_time: str, snapshot: Mapping[str, object]) -> None:
        self._db.prediction_runs.update_one(
            {"owner_user_id": owner_user_id, "stay_id": stay_id, "prediction_time": prediction_time},
            {"$set": {**snapshot, "updated_at": _utcnow_iso()}, "$setOnInsert": {"created_at": _utcnow_iso()}},
            upsert=True,
        )

    def list_prediction_runs(self, *, owner_user_id: str, stay_id: str) -> List[Mapping[str, object]]:
        docs = list(
            self._db.prediction_runs.find({"owner_user_id": owner_user_id, "stay_id": stay_id}).sort("prediction_time", 1)
        )
        return [_without_mongo_id(d) for d in docs]
