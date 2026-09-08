#!/usr/bin/env python3
from __future__ import annotations
import hashlib,os,subprocess,sys,urllib.request
from pathlib import Path
R="Anurag9000/Gram-Connect";B="83e527a6d98a31f0c04897acbed1a552d1c5ec1c";S="72ae9bd8e9e34a09e17047e479acf029a69647a3";AC="f899ac190085228cc95f47725a36bfd105822374";AS="82a10e0a6e221dbf90de92d56a8b87d0f270935f";D=Path(__file__).resolve().parent;U=f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{AC}/tools/repo_training_launcher_adapter.py"
def h(x):return hashlib.sha1(f"blob {len(x)}\0".encode()+x).hexdigest()
def main():
 p=D/".training_control"/"repo_training_launcher_adapter.py"
 if not p.is_file() or h(p.read_bytes())!=AS:
  p.parent.mkdir(parents=True,exist_ok=True);x=urllib.request.urlopen(U,timeout=60).read()
  if h(x)!=AS:raise RuntimeError("Pinned launcher adapter checksum mismatch")
  t=p.with_suffix(".tmp");t.write_bytes(x);os.replace(t,p)
 e=os.environ.copy();e["TRAINING_LAUNCHER_BASE_REPOSITORY"]=R;e["TRAINING_LAUNCHER_BASE_COMMIT"]=B;e["TRAINING_LAUNCHER_BASE_BLOB"]=S;e["TRAINING_CONTROL_REPO_ROOT"]=str(D);return subprocess.call([sys.executable,str(p),*sys.argv[1:]],cwd=D,env=e)
if __name__=="__main__":raise SystemExit(main())
