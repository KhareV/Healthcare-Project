#!/usr/bin/env python3
"""Build Phase-4 engineering or final retained cohort artifacts."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from data.synthetic.cohort_manifest import build_to_directory

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source-manifest",type=Path,required=True); parser.add_argument("--config",type=Path,default=Path("configs/synthetic/cohort_spec_v2.yaml")); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--mode",choices=("engineering","final"),required=True); parser.add_argument("--generation-timestamp")
    args=parser.parse_args(); resolve=lambda p:p if p.is_absolute() else ROOT/p; source,config,output=map(resolve,(args.source_manifest,args.config,args.output))
    try: source_ref=str(source.resolve().relative_to(ROOT)); config_ref=str(config.resolve().relative_to(ROOT)); output_ref=str(output.resolve().relative_to(ROOT))
    except ValueError: parser.error("source manifest, config, and output must resolve inside the repository")
    command=f"python3 scripts/build_synthetic_cohort.py --source-manifest {source_ref} --config {config_ref} --output {output_ref} --mode {args.mode}"
    if args.generation_timestamp: command+=f" --generation-timestamp {args.generation_timestamp}"
    result=build_to_directory(source,config,output,ROOT,mode=args.mode,generation_command=command,generation_timestamp=args.generation_timestamp); print(result.relative_to(ROOT))
if __name__=="__main__": main()
