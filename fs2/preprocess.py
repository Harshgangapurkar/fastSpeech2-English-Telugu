"""Deterministic, audited splits and acoustic features. Alignment is a separate stage."""
import argparse
import hashlib
import random
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from .common import config, encode, vocabulary, read_json, write_json, normalize


def select(rows, train_seconds, val_seconds, seed):
    # Duplicate recordings are removed; all recordings of the same normalized
    # transcript remain together, retaining both Telugu voices without text leakage.
    rows = sorted(rows, key=lambda r:r['id'])
    seen_audio, by_text = set(), {}
    for row in rows:
        key = ' '.join(normalize(row['text']).split())
        if row['audio_hash'] in seen_audio: continue
        seen_audio.add(row['audio_hash']); by_text.setdefault(key, []).append(row)
    groups = list(by_text.values()); random.Random(seed).shuffle(groups)
    splits, cursor = [], 0
    for target in [train_seconds, val_seconds]:
        chosen, total = [], 0.0
        while cursor < len(groups) and total < target:
            group = groups[cursor]; cursor += 1
            chosen.extend(group); total += sum(r['seconds'] for r in group)
        if total < target: raise ValueError(f'Not enough unique usable audio: target {target/3600:.3f}h, available {total/3600:.3f}h for this split')
        splits.append(chosen)
    return splits



def features(path, a):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if not np.isfinite(y).all() or len(y) == 0: raise ValueError('Empty/non-finite audio')
    y = librosa.resample(y, orig_sr=sr, target_sr=a['sample_rate'])
    if np.max(np.abs(y)) < 1e-6: raise ValueError('Silent audio')
    stft = librosa.stft(y, n_fft=a['n_fft'], hop_length=a['hop_length'], win_length=a['win_length'])
    power = np.abs(stft)**2
    basis = librosa.filters.mel(sr=a['sample_rate'], n_fft=a['n_fft'], n_mels=a['n_mels'], fmin=a['fmin'], fmax=a['fmax'])
    mel = np.log(np.maximum(basis @ power, 1e-5)).T.astype('float32')
    f0, voiced, _ = librosa.pyin(y, sr=a['sample_rate'], fmin=a['pitch_fmin'], fmax=a['pitch_fmax'], frame_length=a['n_fft'], hop_length=a['hop_length'])
    pitch = np.nan_to_num(f0, nan=0).astype('float32')
    energy = librosa.feature.rms(S=np.abs(stft), frame_length=a['n_fft'], hop_length=a['hop_length'])[0].astype('float32')
    if not (len(mel)==len(pitch)==len(energy)): raise ValueError('Feature time axes differ')
    return mel, pitch, energy


def prepare(c, manifest):
    out = Path(c['data_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists(): raise FileExistsError(f'{out} already prepared; use a new data_dir to avoid stale features/checkpoints')
    manifest = Path(manifest).resolve(); rows = read_json(manifest)
    vocab = vocabulary(c['language']); valid=[]; rejected=[]; ids=set()
    for raw in rows:
        row=dict(raw)
        if row['id'] in ids or not row['id'].replace('_','').replace('-','').isalnum():
            raise ValueError('IDs must be unique and contain only letters, digits, underscore or hyphen')
        ids.add(row['id'])
        try:
            encode(row['text'],vocab)
            path = (manifest.parent / row['audio']).resolve()
            y,sr=sf.read(path,dtype='float32',always_2d=True)
            seconds=len(y)/sr
            if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-6: raise ValueError('Invalid/silent waveform')
            if not c['selection']['min_seconds'] <= seconds <= c['selection']['max_seconds']: raise ValueError('Duration outside configured range')
            row.update(audio=str(path),seconds=seconds,audio_hash=hashlib.sha256(str(sr).encode()+y.tobytes()).hexdigest(),speaker=str(row.get('speaker','pooled')))
            valid.append(row)
        except (ValueError, RuntimeError) as e: rejected.append({'id':row['id'],'reason':str(e)})
    write_json(out/'rejected.json',rejected)
    train,val=select(valid,c['selection']['train_hours']*3600,c['selection']['validation_hours']*3600,c['seed'])
    for selected,target in [(train,c['selection']['train_hours']*3600),(val,c['selection']['validation_hours']*3600)]:
        if sum(r['seconds'] for r in selected)-target > c['selection']['tolerance_seconds']:
            raise ValueError('Transcript group overshoots duration tolerance; choose another seed')
    speakers={s:i for i,s in enumerate(sorted({r['speaker'] for r in train}))}
    unseen={r['speaker'] for r in val}-set(speakers)
    if unseen: raise ValueError(f'Validation has unseen speaker groups {unseen}; change seed or supply a suitable manifest')
    write_json(out/'vocab.json',vocab); write_json(out/'speakers.json',speakers)
    (out/'features').mkdir(exist_ok=True)
    for row in train+val:
        mel,pitch,energy=features(row['audio'],c['audio']); tokens=encode(row['text'],vocab)
        if len(mel)>c['model']['max_seq_len'] or len(tokens)>c['model']['max_seq_len']: raise ValueError(f"{row['id']}: sequence too long")
        if len(mel)<len(tokens)+int((tokens[1:]==tokens[:-1]).sum()): raise ValueError(f"{row['id']}: impossible CTC alignment")
        np.savez_compressed(out/'features'/f"{row['id']}.npz",mel=mel,pitch=pitch,energy=energy,tokens=tokens,speaker=np.array(speakers[row['speaker']]))
        print(f"features {row['id']}",flush=True)
    write_json(out/'train.json',train);write_json(out/'validation.json',val)
    metadata={'language':c['language'],'dataset':c['dataset'],'seed':c['seed'],'audio':c['audio'],'vocab':vocab,'speakers':speakers,'train_ids':[r['id'] for r in train],'validation_ids':[r['id'] for r in val],'audio_hashes':{r['id']:r['audio_hash'] for r in train+val},'train_hours':sum(r['seconds'] for r in train)/3600,'validation_hours':sum(r['seconds'] for r in val)/3600,'split_policy':'seeded transcript-group split; deduplicated audio; speakers may overlap','rejected_count':len(rejected)}
    write_json(out/'metadata.json',metadata)
    print('Prepared:',metadata['train_hours'],'train hours;',metadata['validation_hours'],'validation hours')


def main(default):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=default);p.add_argument('--manifest',required=True)
    a=p.parse_args();prepare(config(a.config),a.manifest)
