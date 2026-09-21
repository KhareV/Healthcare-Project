#!/usr/bin/env python3
import argparse
from pathlib import Path
from labels.phase9_final import build
def main():
    p=argparse.ArgumentParser()
    for x in ("cohort-manifest","processed-manifest","feature-manifest","output"): p.add_argument("--"+x,type=Path,required=True)
    p.add_argument("--generation-timestamp",default="2026-09-21T00:00:00Z"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    print(build(cohort_manifest_path=root/a.cohort_manifest,processed_manifest_path=root/a.processed_manifest,feature_manifest_path=root/a.feature_manifest,output=root/a.output,root=root,generation_timestamp=a.generation_timestamp))
if __name__=="__main__": main()
