"""Audit the six synthetic exact-30 manifests without training or test access."""

import argparse
import json
from pathlib import Path

from experiments.audit import audit_search
from experiments.search_governance import load_candidate_manifest, load_search_space


def run_dry_audit(root=Path(".")):
    manifest_paths = sorted((root / "artifacts/search/dry_run").glob("*_v1.json"))
    summaries = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        space, digest = load_search_space(root / manifest["search_space_ref"])
        if digest != manifest["search_space_hash"]:
            raise ValueError("search-space hash mismatch")
        _, candidates = load_candidate_manifest(path, space)
        summaries.append(audit_search(manifest, candidates))
    if len(summaries) != 6:
        raise ValueError("dry-run audit requires six family/task manifests")
    return {"searches": summaries, "scientific_searches_executed": 0, "test_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    print(json.dumps(run_dry_audit(arguments.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
