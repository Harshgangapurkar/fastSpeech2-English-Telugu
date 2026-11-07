"""Configuration, text inventory and reproducibility helpers."""
import hashlib
import json
import random
import unicodedata
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def config(path):
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ['data_dir', 'results_dir', 'checkpoint_dir']:
        c[key] = str((ROOT / c[key]).resolve())
    return c

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def device(name='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)

def normalize(text):
    # No spelling correction, number expansion, transliteration or whitespace repair.
    return unicodedata.normalize('NFC', text).lower()

def vocabulary(language):
    # 0 is padding/CTC blank. Unsupported characters are errors, never silently dropped.
    chars = [chr(i) for i in range(32, 127)]
    chars += ['’', '‘', '“', '”', '–', '—', '…', '\t', '\n', '\r', '\u200c', '\u200d']
    if language == 'Telugu': chars += [chr(i) for i in range(0x0C00, 0x0C80)]
    return {ch: i+1 for i, ch in enumerate(chars)}

def encode(text, vocab):
    value = normalize(text)
    missing = sorted(set(value) - set(vocab))
    if missing: raise ValueError(f'Unsupported characters {missing!r} in {text!r}; review transcription/inventory.')
    if not value.strip(): raise ValueError('Empty transcript')
    return np.array([vocab[ch] for ch in value], dtype=np.int64)

def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def fingerprint(c, metadata):
    # Bind weights to exact features, vocabulary, split and architecture, not absolute paths.
    return hashlib.sha256(json.dumps({'audio':c['audio'],'model':c['model'],'metadata':metadata}, sort_keys=True).encode()).hexdigest()
