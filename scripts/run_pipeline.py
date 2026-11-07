"""Run all stages for one language; do not rerun into already prepared data directories."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--language',choices=['English','Telugu'],required=True);p.add_argument('--device',default='auto');p.add_argument('--manifest',help='Use local data instead of downloading');p.add_argument('--accept-indictts-license',action='store_true');a=p.parse_args()
def run(args): subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
lang=a.language
if not a.manifest:
    args=['scripts/download_data.py','--language',lang]
    if a.accept_indictts_license:args+=['--accept-indictts-license']
    run(args)
manifest=str(Path(a.manifest).resolve()) if a.manifest else f'data/raw/{lang}/manifest.json'
run([f'{lang}/preprocess.py','--manifest',manifest]);run(['-m','fs2.alignment','--config',f'{lang}/config.yaml','--device',a.device]);run([f'{lang}/train.py','--device',a.device]);run([f'{lang}/inference.py','--device',a.device]);run(['scripts/build_report.py'])
