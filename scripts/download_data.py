"""Export the assignment's HF datasets into the common local manifest format."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fs2.common import ROOT,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--language',required=True,choices=['English','Telugu']);p.add_argument('--revision',default='main');p.add_argument('--accept-indictts-license',action='store_true');p.add_argument('--limit',type=int,help='Debug only: cannot meet six-hour requirement')
    a=p.parse_args()
    if a.language=='Telugu' and not a.accept_indictts_license: p.error('Read https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf and pass --accept-indictts-license if you accept its terms')
    from datasets import load_dataset,Audio
    import soundfile as sf
    lang=a.language;repo='ai4bharat/Svarah' if lang=='English' else 'SPRINGLab/IndicTTS_Telugu';split='test' if lang=='English' else 'train';column='audio_filepath' if lang=='English' else 'audio'
    data=load_dataset(repo,split=split,revision=a.revision).cast_column(column,Audio())
    dest=ROOT/'data'/'raw'/lang;(dest/'wav').mkdir(parents=True,exist_ok=True);rows=[]
    for index,row in enumerate(data):
        if a.limit and index>=a.limit: break
        id=f'{lang.lower()}_{index:06d}';wav=row[column]
        # Svarah exposes demographics, not a verified speaker ID: pooled voice is honest.
        # Telugu card documents one male and one female speaker, so gender identifies these two.
        speaker='pooled' if lang=='English' else f"telugu_{row['gender']}"
        sf.write(dest/'wav'/f'{id}.wav',wav['array'],wav['sampling_rate'],subtype='PCM_16')
        rows.append({'id':id,'audio':f'wav/{id}.wav','text':row['text'],'speaker':speaker})
        if index%100==0: print(f'exported {index}',flush=True)
    write_json(dest/'manifest.json',rows)
    write_json(dest/'source.json',{'repository':repo,'revision_requested':a.revision,'datasets_fingerprint':data._fingerprint,'source_split':split,'exported_rows':len(rows),'note':'Source split is repurposed for this assignment, not a valid Svarah ASR benchmark evaluation.'})
    print(dest/'manifest.json')
if __name__=='__main__':main()
