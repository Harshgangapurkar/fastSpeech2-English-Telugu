"""Fail visibly when mandatory empirical deliverables are absent."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fs2.common import ROOT, read_json
missing=[]
for lang in ['English','Telugu']:
    paths=[ROOT/lang/'checkpoints/best.pt',ROOT/'Results'/lang/'training_loss.png',ROOT/'Results'/lang/'validation_loss.png',ROOT/'Results'/lang/'loss_history.json',ROOT/'data'/lang/'metadata.json',ROOT/'data'/lang/'stats.json',ROOT/'data'/lang/'vocab.json',ROOT/'data'/lang/'speakers.json']
    for r in read_json(ROOT/lang/'test_sentences.json'):
        paths += [ROOT/'Results'/lang/'test_generated_audios'/f"{r['id']}.wav"]
        paths += [ROOT/'Results'/lang/'variance_adaptor_plots'/v/f"{r['id']}.png" for v in ['duration','pitch','energy']]
    missing += [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    meta=ROOT/'data'/lang/'metadata.json'
    if meta.exists():
        m=read_json(meta)
        for key,target in [('train_hours',5),('validation_hours',1)]:
            if not target<=m[key]<=target+60/3600:missing.append(f'{lang}: {key} outside required range')
notes=read_json(ROOT/'Report/listening_notes.json')
for id,row in notes.items():
    if any(not str(v).strip() or str(v).startswith('NOT EVALUATED') for v in row.values()):missing.append(f'Listening/variance analysis: {id}')
if not (ROOT/'Report/Technical_Report.pdf').exists():missing.append('Report/Technical_Report.pdf')
if missing:
    print('NOT READY FOR SUBMISSION. Missing or incomplete:');print('\n'.join('- '+x for x in missing));sys.exit(1)
print('Expected files and notes present. Manually check report freshness, actual speech quality, split audit, and public repository/download accessibility before submission.')
