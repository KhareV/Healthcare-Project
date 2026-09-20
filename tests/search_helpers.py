import json
from pathlib import Path

from experiments.search_governance import load_candidate_manifest, load_search_space


ROOT = Path(__file__).resolve().parents[1]


def manifest_paths():
    return tuple(sorted((ROOT / "artifacts/search/dry_run").glob("*_v1.json")))


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    space, digest = load_search_space(ROOT / manifest["search_space_ref"])
    assert digest == manifest["search_space_hash"]
    _, candidates = load_candidate_manifest(path, space)
    return manifest, space, candidates
