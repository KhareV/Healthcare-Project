#!/usr/bin/env python3
import argparse
from pathlib import Path
from labels.synthetic_manifest import build_partial_target_bundle

def main():
    p=argparse.ArgumentParser()
    for name in ("cohort-manifest","processed-manifest","feature-manifest","recovery-spec","icu-spec","target-schema","target-provenance","sofa-spec","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=("engineering","final"),required=True); p.add_argument("--generation-timestamp")
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    print(build_partial_target_bundle(cohort_manifest_path=a.cohort_manifest,processed_manifest_path=a.processed_manifest,
        feature_manifest_path=a.feature_manifest,recovery_spec_path=a.recovery_spec,icu_spec_path=a.icu_spec,
        target_schema_path=a.target_schema,target_provenance_path=a.target_provenance,sofa_spec_path=a.sofa_spec,output=a.output,root=root,mode=a.mode,
        generation_timestamp=a.generation_timestamp))
if __name__=="__main__": main()
