import numpy as np, torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
AA='ACDEFGHIKLMNPQRSTVWY'
KD=dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
def hm(s,ang=100):
    r=np.deg2rad(ang)*np.arange(len(s)); h=np.array([KD[a] for a in s])
    return np.hypot((h*np.cos(r)).sum(),(h*np.sin(r)).sum())/len(s)
def phys1(s):
    pa=ProteinAnalysis(s); ch=s.count('K')+s.count('R')-s.count('D')-s.count('E')+0.1*s.count('H')
    comp=pa.get_amino_acids_percent()
    return [len(s),ch,ch/len(s),pa.gravy(),pa.isoelectric_point(),pa.aromaticity(),pa.instability_index(),hm(s)]+[comp[a] for a in AA]
def phys(seqs): return np.array([phys1(s) for s in seqs])
ESM={'esm8':'facebook/esm2_t6_8M_UR50D','esm35':'facebook/esm2_t12_35M_UR50D','esm150':'facebook/esm2_t30_150M_UR50D'}
def embed(seqs,key,bs=64):
    from transformers import AutoTokenizer, EsmModel
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(ESM[key]); m=EsmModel.from_pretrained(ESM[key]).to(dev).eval(); out=[]
    for i in range(0,len(seqs),bs):
        b=tok(list(seqs[i:i+bs]),return_tensors='pt',padding=True).to(dev)
        with torch.no_grad(): h=m(**b).last_hidden_state
        mask=b['attention_mask'].clone(); L=mask.sum(1); mask[:,0]=0; mask[torch.arange(len(L)),L-1]=0
        mask=mask.unsqueeze(-1).float(); out.append(((h*mask).sum(1)/mask.sum(1)).cpu().numpy())
    return np.vstack(out)
