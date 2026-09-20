#!/usr/bin/env python3
"""Generate Phase-3 fixture/smoke raw artifacts; final mode fails closed."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from data.synthetic.generator import generate_to_directory
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--generation-timestamp")
    args=parser.parse_args(); config=args.config if args.config.is_absolute() else ROOT/args.config; output=args.output if args.output.is_absolute() else ROOT/args.output
    try:
        config_ref=str(config.resolve().relative_to(ROOT)); output_ref=str(output.resolve().relative_to(ROOT))
    except ValueError as error:
        parser.error("config and output must resolve inside the repository for portable provenance")
    command="python3 scripts/generate_synthetic_dataset.py --config "+config_ref+" --output "+output_ref
    if args.generation_timestamp: command+=" --generation-timestamp "+args.generation_timestamp
    print(generate_to_directory(config,output,ROOT,command,args.generation_timestamp).relative_to(ROOT))
if __name__=="__main__": main()
