#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
REPOSITORY="Anurag9000/Gram-Connect";COMMIT="a3d25bd282d8e2e0d9f2cb3730d4d036703a39ad";SHA="2515a8a2fc2dbed7334a4240438dc095f19636d2";URL=f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{COMMIT}/tools/universal_training_controller_entry.py";ROOT=Path(__file__).resolve().parent
PROFILE={"repository":REPOSITORY,"preferred_training_entrypoints":["train.py","training.py","run_training.py","scripts/train.py","scripts/train_all.py","scripts/run_training.py"],"preferred_dataset_entrypoints":["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/materialize_datasets.py","scripts/dataset_setup.py"],"dynamic_registry_covers":[],"extra_jobs":[],"ignore_entrypoints":["run_all_training.py"],"strict_coverage":True,"require_native_resume":True,"require_exact_resume":True,"require_training_exact_resume":True,"require_training_early_stopping":True,"require_dag_enforcement":True,"require_model_surface_accounting":True,"require_literal_opf_mechanism_parity":True,"require_well_formed_training_exemptions":True}
def h(b):return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
def main():
 c=ROOT/".training_control"/"universal_training_controller_entry.py"
 if not c.is_file() or h(c.read_bytes())!=SHA:
  c.parent.mkdir(parents=True,exist_ok=True);d=urllib.request.urlopen(URL,timeout=60).read()
  if h(d)!=SHA:raise RuntimeError("Pinned training controller checksum mismatch")
  t=c.with_suffix(".tmp");t.write_bytes(d);os.replace(t,c)
 e=os.environ.copy();e["TRAINING_CONTROL_PROFILE"]=json.dumps(PROFILE,separators=(",",":"));e["TRAINING_CONTROL_REPO_ROOT"]=str(ROOT);e.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC","30");return subprocess.call([sys.executable,str(c),*sys.argv[1:]],cwd=ROOT,env=e)
if __name__=="__main__":raise SystemExit(main())