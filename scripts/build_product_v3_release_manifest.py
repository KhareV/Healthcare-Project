#!/usr/bin/env python3
"""Build a product-layer V3 release record from verified, existing evidence.

This script reads scientific files and never writes to their directories. Test evidence
must be supplied as JSON, so no pass count is invented by the release builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/product_v3'
INVENTORY = OUT / 'governance/pre_phase2_scientific_inventory_v1.sha256'
EXPECTED_SELECTED = 'af9014a33d0a219a3f64b713c6812add46b07f44efcb4d212674ea47b7011961'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reference(name: str) -> dict:
    path = ROOT / name
    if not path.is_file():
        raise FileNotFoundError(name)
    return {'path': name, 'sha256': digest(path)}


def verify_science() -> dict:
    mismatches = []
    checked = 0
    for line in INVENTORY.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split('  ', 1)
        path = ROOT / relative
        checked += 1
        if not path.is_file() or digest(path) != expected:
            mismatches.append(relative)
    selected = digest(ROOT / 'artifacts/performance_v3/phase3/selected_models_v3.json')
    if selected != EXPECTED_SELECTED:
        mismatches.append('artifacts/performance_v3/phase3/selected_models_v3.json (handoff mismatch)')
    ledger = json.loads((ROOT / 'artifacts/performance_v3/governance/v3_final_test_access_state.json').read_text())
    if ledger.get('state') != 'FINAL_RUN_COMPLETED' and ledger.get('status') != 'FINAL_RUN_COMPLETED':
        mismatches.append('artifacts/performance_v3/governance/v3_final_test_access_state.json (state)')
    return {'status': 'PASS' if not mismatches else 'FAIL', 'files_checked': checked,
            'mismatches': mismatches, 'selected_models_v3_sha256': selected,
            'v3_final_test_ledger_state': ledger.get('state', ledger.get('status')),
            'inventory_sha256': digest(INVENTORY)}


def git(*args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-science', action='store_true', help='read-only full scientific hash audit')
    parser.add_argument('--verification', type=Path, help='JSON with actual backend/frontend/browser test results')
    args = parser.parse_args()
    science = verify_science()
    if args.verify_science:
        print(json.dumps(science, indent=2))
        if science['status'] != 'PASS':
            raise SystemExit(1)
        return
    if science['status'] != 'PASS':
        raise SystemExit(f"Scientific immutability failed: {science['mismatches']}")
    handoff = json.loads((ROOT / 'artifacts/performance_v3_1/product_handoff.json').read_text())
    if handoff['outcome'] != 'V3_RETAINED_AFTER_NEGATIVE_V3_1_DEVELOPMENT' or handoff['v3_1_support_candidate']['serve']:
        raise SystemExit('Unexpected V3.1 handoff decision')
    if handoff['effective_release']['manifest_sha256'] != EXPECTED_SELECTED:
        raise SystemExit('V3.1 handoff disagrees with selected bundle')
    registry = json.loads((ROOT / 'artifacts/performance_v3/research/research_registry_v1.json').read_text())
    screenshots = sorted((ROOT / 'docs/evidence/product_v3').glob('*.png'))
    verification = json.loads(args.verification.read_text()) if args.verification else None
    status = 'READY' if verification and verification.get('all_required_passed') is True else 'INTEGRATION_EVIDENCE_PENDING'
    record = {
        'artifact_version': 'product_v3_release_manifest_v1',
        'kind': 'ENGINEERING_PRODUCT_RELEASE_NOT_SCIENTIFIC_FREEZE',
        'status': status,
        'source_code_commit': git('rev-parse', 'HEAD'),
        'worktree_dirty_at_generation': bool(git('status', '--porcelain')),
        'scientific_release': {'v3_1_handoff': reference('artifacts/performance_v3_1/product_handoff.json'),
                               'effective_bundle': reference('artifacts/performance_v3/phase3/selected_models_v3.json'),
                               'v3_final_evaluation_freeze': reference('artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json'),
                               'v3_final_metrics': reference('artifacts/performance_v3/phase4/final_metrics_v3.json'),
                               'v3_test_access_ledger': reference('artifacts/performance_v3/governance/v3_final_test_access_state.json'),
                               'v3_1_candidate_served': False},
        'product': {'api_entry_point': 'api.product_app:app --factory',
                    'prediction_schema_version': 'v3_prediction_response_v1',
                    'serving_version': 'performance_v3_serving_v1',
                    'frontend_version': json.loads((ROOT / 'frontend/package.json').read_text())['version'],
                    'patient_record_schema': 'encounter_v2',
                    'prediction_history_identity': 'owner+encounter+cutoff+bundle+data_revision',
                    'report_intelligence_version': 'report_intelligence_v1',
                    'chart_export_version': registry['export_version'],
                    'showcase_manifest': reference('configs/performance_v3_product/showcase_manifest_v1.json'),
                    'research_registry': reference('artifacts/performance_v3/research/research_registry_v1.json'),
                    'research_export_count': len(registry['exports']),
                    'screenshot_evidence': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in screenshots]},
        'documentation': [reference(x) for x in ('README.md', 'RUNBOOK.md',
                         'docs/product_v3/PRODUCT_ARCHITECTURE.md',
                         'docs/product_v3/FINAL_DEMO_GUIDE.md',
                         'docs/product_v3/FIGURE_AND_TABLE_INDEX.md',
                         'docs/product_v3/LIMITATIONS.md')],
        'verification': verification,
        'scientific_immutability': science,
        'release_builder_final_test_reinference_performed': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'product_release_manifest_v1.json'
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(path.relative_to(ROOT), digest(path), status)


if __name__ == '__main__':
    main()
