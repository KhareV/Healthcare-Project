#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from data.synthetic.provenance import sha256_file
from integration.phase14 import HANDOFF_PATH, pulkit_receiver_acceptance, validate_handoff, vedant_receiver_acceptance
def main():
    path=ROOT/HANDOFF_PATH; handoff=validate_handoff(ROOT,path)
    vedant=vedant_receiver_acceptance(ROOT,path); pulkit=pulkit_receiver_acceptance(ROOT,path)
    status=json.loads((ROOT/"artifacts/handoffs/sanskruti_phase14_status_v1.json").read_text())
    if status["handoff_sha256"]!=sha256_file(path) or status["test_accessed"] is not False: raise RuntimeError("Phase-14 status lineage mismatch")
    print(json.dumps({"status":status["status"],"g1_sha256":handoff["g1_sha256"],"vedant":vedant["status"],"pulkit":pulkit["status"],"test_accessed":False},sort_keys=True))
if __name__=="__main__": main()
