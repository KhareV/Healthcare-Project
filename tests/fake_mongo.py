"""Minimal in-memory stand-in for the subset of pymongo used by serving.v2.persistence (test helper only)."""
import copy
import itertools

_ids = itertools.count(1)


def _match(doc, flt):
    for key, cond in flt.items():
        value = doc.get(key)
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, arg in cond.items():
                if op == "$exists" and (key in doc) != bool(arg):
                    return False
                if op == "$in" and value not in arg:
                    return False
                if op not in ("$exists", "$in"):
                    raise NotImplementedError(op)
        elif value != cond:
            return False
    return True


class _Cursor(list):
    def sort(self, key, direction=1):
        self.sort_by = key
        super().sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=direction < 0)
        return self


class FakeCollection:
    def __init__(self):
        self.docs = []

    def create_index(self, *a, **k):
        return "idx"

    def find_one(self, flt=None):
        for d in self.docs:
            if _match(d, flt or {}):
                return copy.deepcopy(d)
        return None

    def find(self, flt=None):
        return _Cursor(copy.deepcopy(d) for d in self.docs if _match(d, flt or {}))

    def insert_one(self, doc):
        d = copy.deepcopy(doc); d.setdefault("_id", next(_ids)); self.docs.append(d)

    def delete_one(self, flt):
        for i, d in enumerate(self.docs):
            if _match(d, flt):
                del self.docs[i]
                class R: deleted_count = 1
                return R
        class R0: deleted_count = 0
        return R0

    def update_one(self, flt, update, upsert=False, array_filters=None):
        target = next((d for d in self.docs if _match(d, flt)), None)
        inserted = False
        if target is None:
            if not upsert:
                class R0: matched_count = 0; modified_count = 0
                return R0
            target = {"_id": next(_ids)}
            for k, v in flt.items():
                if not (isinstance(v, dict) and any(x.startswith("$") for x in v)):
                    target[k] = v
            self.docs.append(target); inserted = True
        for k, v in update.get("$set", {}).items():
            target[k] = copy.deepcopy(v)
        if inserted:
            for k, v in update.get("$setOnInsert", {}).items():
                target[k] = copy.deepcopy(v)
        for k, v in update.get("$push", {}).items():
            target.setdefault(k, []).extend(copy.deepcopy(v["$each"]) if isinstance(v, dict) else [copy.deepcopy(v)])
        for k, v in update.get("$inc", {}).items():
            target[k] = target.get(k, 0) + v
        class R: matched_count = 1; modified_count = 1
        return R


class FakeDb:
    def __init__(self):
        self._c = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._c.setdefault(name, FakeCollection())


class FakeClient:
    def __init__(self):
        self._dbs = {}

    def __getitem__(self, name):
        return self._dbs.setdefault(name, FakeDb())


def fake_persistence():
    from serving.v2.persistence import MongoPersistence

    p = MongoPersistence(None)
    p._client = FakeClient()
    p.enabled = True
    return p
