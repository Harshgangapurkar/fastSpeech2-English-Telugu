"""Train-only CTC aligner and monotonic Viterbi duration extraction (no uniform targets)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .common import config, seed_all, device, read_json, write_json, fingerprint

class Aligner(nn.Module):
    def __init__(self,vocab_size,hidden):
        super().__init__();self.rnn=nn.LSTM(80,hidden,num_layers=2,batch_first=True,bidirectional=True,dropout=0.1);self.proj=nn.Linear(2*hidden,vocab_size)
    def forward(self,x,lengths):
        # Fixed scaling, not validation-derived statistics.
        x=pack_padded_sequence((x+5)/5,lengths.cpu(),batch_first=True,enforce_sorted=False)
        x,_=self.rnn(x);x,_=pad_packed_sequence(x,batch_first=True)
        return self.proj(x).log_softmax(-1)


def viterbi(logp,tokens):
    """CTC state path with legal blank/repeated-label transitions; cover every frame."""
    tokens=np.asarray(tokens,dtype=np.int64); T,V=logp.shape; N=len(tokens)
    if N==0 or T<N+int((tokens[1:]==tokens[:-1]).sum()): raise ValueError('Impossible CTC alignment')
    labels=np.zeros(2*N+1,dtype=np.int64);labels[1::2]=tokens; S=len(labels)
    prev=np.full(S,-np.inf);prev[0]=logp[0,0];prev[1]=logp[0,tokens[0]]
    back=np.zeros((T,S),dtype=np.int8)
    skip=(labels!=0)&(labels!=np.roll(labels,2));skip[:2]=False
    for t in range(1,T):
        one=np.r_[-np.inf,prev[:-1]];two=np.r_[[-np.inf,-np.inf],prev[:-2]];two[~skip]=-np.inf
        candidates=np.stack([prev,one,two]);which=candidates.argmax(0)
        prev=candidates[which,np.arange(S)]+logp[t,labels];back[t]=which
    state=S-1 if prev[-1]>prev[-2] else S-2
    if not np.isfinite(prev[state]): raise ValueError('No finite CTC path')
    score=float(prev[state]/T);states=np.zeros(T,dtype=np.int64)
    for t in range(T-1,-1,-1): states[t]=state;state-=back[t,state]
    # Interior blanks belong to the preceding grapheme; leading blanks to the first.
    owners=np.clip((states-1)//2,0,N-1);dur=np.bincount(owners,minlength=N)
    if dur.sum()!=T or (dur<1).any(): raise AssertionError('Invalid alignment coverage')
    return dur,score


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1): nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def greedy(logp):
    ids=logp.argmax(-1).tolist();return [v for i,v in enumerate(ids) if v and (i==0 or ids[i-1]!=v)]


def load_item(root,row):
    with np.load(root/'features'/f"{row['id']}.npz") as z: return {k:z[k] for k in z.files}


def fit_and_extract(c,dev,epochs=None):
    seed_all(c['seed']);root=Path(c['data_dir']);meta=read_json(root/'metadata.json')
    train=read_json(root/'train.json');val=read_json(root/'validation.json');ac=c['alignment']
    model=Aligner(len(read_json(root/'vocab.json'))+1,ac['hidden']).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=ac['lr']);lossfn=nn.CTCLoss(blank=0,zero_infinity=False)
    ck=root/'aligner.pt';history=[];best=float('inf')
    for epoch in range(1,(epochs or ac['epochs'])+1):
        model.train();order=np.random.permutation(len(train));total=0;count=0
        for start in range(0,len(order),ac['batch_size']):
            items=[load_item(root,train[int(i)]) for i in order[start:start+ac['batch_size']]]
            lengths=torch.tensor([len(x['mel']) for x in items]);target_lens=torch.tensor([len(x['tokens']) for x in items])
            x=pad_sequence([torch.from_numpy(i['mel']) for i in items],batch_first=True).to(dev)
            targets=torch.cat([torch.from_numpy(i['tokens']) for i in items]).to(dev)
            loss=lossfn(model(x,lengths).transpose(0,1),targets,lengths,target_lens)
            if not torch.isfinite(loss): raise ValueError('Nonfinite CTC loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(items);count+=len(items)
        model.eval();errors=chars=0
        with torch.no_grad():
            # Fixed train-only diagnostic subset; validation never fits/selects aligner.
            for row in train[:min(64,len(train))]:
                i=load_item(root,row);lp=model(torch.from_numpy(i['mel'])[None].to(dev),torch.tensor([len(i['mel'])]))[0]
                errors+=distance(greedy(lp),i['tokens'].tolist());chars+=len(i['tokens'])
        cer=errors/chars;history.append({'epoch':epoch,'ctc_loss':total/count,'train_subset_cer':cer})
        write_json(root/'alignment_training.json',history)
        print(f'align epoch={epoch} loss={total/count:.4f} train_subset_CER={cer:.3f}',flush=True)
        if cer<best:
            best=cer;torch.save({'model':model.state_dict(),'fingerprint':fingerprint(c,meta),'cer':cer},ck)
    state=torch.load(ck,map_location=dev,weights_only=False);model.load_state_dict(state['model'])
    if state['cer']>ac['max_train_cer']:
        raise RuntimeError(f"Aligner CER {state['cer']:.3f} exceeds gate {ac['max_train_cer']}; increase alignment epochs and inspect transcripts. No durations exported.")
    model.eval();diagnostics=[];raw={}
    with torch.no_grad():
        for row in train+val:
            i=load_item(root,row);lp=model(torch.from_numpy(i['mel'])[None].to(dev),torch.tensor([len(i['mel'])]))[0].cpu().numpy()
            dur,score=viterbi(lp,i['tokens']);p=[];e=[];offset=0
            for n in dur:
                f=i['pitch'][offset:offset+n];voiced=f[f>0]
                p.append(float(voiced.mean()) if len(voiced) else 0.0);e.append(float(i['energy'][offset:offset+n].mean()));offset+=n
            i.update(duration=dur,pitch=np.array(p,dtype='float32'),energy=np.array(e,dtype='float32'));raw[row['id']]=i
            diagnostics.append({'id':row['id'],'split':'train' if row in train else 'validation','mean_log_probability':score,'cer':distance(greedy(torch.from_numpy(lp)),i['tokens'].tolist())/len(i['tokens']),'max_token_frames':int(dur.max()),'frames':int(dur.sum())})
    stats={}
    for key in ['pitch','energy']:
        values=np.concatenate([raw[r['id']][key] for r in train]);mean=float(values.mean());std=max(float(values.std()),1e-6)
        norm=(values-mean)/std;stats[key]=[float(norm.min()),float(norm.max()),mean,std]
    write_json(root/'stats.json',stats)
    (root/'aligned').mkdir(exist_ok=True)
    for id,i in raw.items():
        for key in ['pitch','energy']: i[key]=(i[key]-stats[key][2])/stats[key][3]
        np.savez_compressed(root/'aligned'/f'{id}.npz',**i)
    write_json(root/'alignment_diagnostics.json',diagnostics)
    write_json(root/'alignment_metadata.json',{'fingerprint':fingerprint(c,meta),'stats':stats,'method':'train-only bidirectional LSTM CTC + Viterbi; blanks assigned to preceding token','best_train_subset_cer':best})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int)
    a=p.parse_args();fit_and_extract(config(a.config),device(a.device),a.epochs)
if __name__=='__main__': main()
