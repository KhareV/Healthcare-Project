#!/usr/bin/env python3
import argparse
from pathlib import Path
from features.manifest import build_feature_bundle

def main():
    p=argparse.ArgumentParser(); p.add_argument("--processed-manifest",type=Path,required=True); p.add_argument("--structural-index",type=Path,required=True); p.add_argument("--feature-schema",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=("engineering","final"),required=True); p.add_argument("--generation-timestamp")
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    result=build_feature_bundle(processed_manifest_path=a.processed_manifest,structural_index_path=a.structural_index,feature_schema_path=a.feature_schema,output=a.output,root=root,mode=a.mode,generation_timestamp=a.generation_timestamp); print(result)
if __name__=="__main__": main()
