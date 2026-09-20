#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from data.synthetic.processed_manifest import build_processed_bundle
def main():
    p=argparse.ArgumentParser(); p.add_argument("--cohort-manifest",type=Path,required=True); p.add_argument("--concept-map",type=Path,default=Path("configs/synthetic/concept_map_v1.json")); p.add_argument("--processed-schema",type=Path,default=Path("configs/synthetic/processed_schema_v1.json")); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=("engineering","final"),required=True); p.add_argument("--generation-timestamp"); a=p.parse_args(); resolve=lambda x:x if x.is_absolute() else ROOT/x; cm,mp,sp,out=map(resolve,(a.cohort_manifest,a.concept_map,a.processed_schema,a.output))
    refs=[]
    for value in (cm,mp,sp,out):
        try: refs.append(str(value.resolve().relative_to(ROOT)))
        except ValueError: p.error("all paths must resolve inside the repository")
    command=f"python3 scripts/build_synthetic_timeline.py --cohort-manifest {refs[0]} --concept-map {refs[1]} --processed-schema {refs[2]} --output {refs[3]} --mode {a.mode}"
    if a.generation_timestamp: command+=f" --generation-timestamp {a.generation_timestamp}"
    print(build_processed_bundle(cm,mp,sp,out,ROOT,mode=a.mode,generation_command=command,generation_timestamp=a.generation_timestamp).relative_to(ROOT))
if __name__=="__main__": main()
