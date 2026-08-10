
"""
Bipartite graph-context assignment scorer for NGFLO Gold Dataset v2.

Architecture:
1. Vehicle-node MLP encoder.
2. Position-node MLP encoder.
3. One bipartite mean-message-passing layer.
4. Global pooled ferry-context vector.
5. Edge MLP scorer using updated vehicle, position, edge, and global features.

The model is intentionally compact. It tests whether global graph context improves
Recall@K beyond the independent-edge logistic baseline.
"""

from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn

DATA = Path("/mnt/data/ngflo_gold_dataset_v2")
OUT = Path("/mnt/data/ngflo_graph_scorer_v2")
OUT.mkdir(exist_ok=True)

SEED=20260810
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)

inst=pd.read_csv(DATA/"instances.csv")
veh=pd.read_csv(DATA/"vehicles.csv")
pos=pd.read_csv(DATA/"positions.csv")
edges=pd.read_csv(DATA/"compatibility_edges.csv")
labs=pd.read_csv(DATA/"assignment_labels.csv")
splits=pd.read_csv(DATA/"splits.csv")

edge_df=edges.merge(labs,on=["instance_id","veh_id","pos_id"],how="inner")

classes=sorted(veh["class"].unique().tolist())
class_to_idx={c:i for i,c in enumerate(classes)}

veh_num=["length","width","height","mass","destination","priority"]
pos_num=["deck","lane","length_cap","width_cap","height_cap","mass_cap","y_coord"]
edge_num=["length_ratio","mass_ratio","abs_y_mass","destination_deck_distance"]

train_ids=set(splits.loc[splits["split"]=="train","instance_id"])
train_veh=veh[veh.instance_id.isin(train_ids)]
train_pos=pos[pos.instance_id.isin(train_ids)]
train_edge=edge_df[edge_df.instance_id.isin(train_ids)]

def stats(df, cols):
    mu=df[cols].mean().to_numpy(np.float32)
    sd=df[cols].std().replace(0,1).fillna(1).to_numpy(np.float32)
    return mu,sd
vmu,vsd=stats(train_veh,veh_num)
pmu,psd=stats(train_pos,pos_num)
emu,esd=stats(train_edge,edge_num)

def make_instance(iid):
    V=veh[veh.instance_id==iid].sort_values("veh_id").copy()
    P=pos[pos.instance_id==iid].sort_values("pos_id").copy()
    E=edge_df[edge_df.instance_id==iid].copy()
    vmap={int(x):i for i,x in enumerate(V.veh_id)}
    pmap={int(x):i for i,x in enumerate(P.pos_id)}
    vn=(V[veh_num].to_numpy(np.float32)-vmu)/vsd
    one=np.zeros((len(V),len(classes)),np.float32)
    for q,c in enumerate(V["class"]): one[q,class_to_idx[c]]=1
    vx=np.concatenate([vn,one],axis=1)
    px=(P[pos_num].to_numpy(np.float32)-pmu)/psd
    ex=(E[edge_num].to_numpy(np.float32)-emu)/esd
    vi=np.array([vmap[int(x)] for x in E.veh_id],dtype=np.int64)
    pi=np.array([pmap[int(x)] for x in E.pos_id],dtype=np.int64)
    y=E.label.to_numpy(np.float32)
    return {
        "iid":iid,
        "vx":torch.tensor(vx),
        "px":torch.tensor(px),
        "ex":torch.tensor(ex),
        "vi":torch.tensor(vi),
        "pi":torch.tensor(pi),
        "y":torch.tensor(y),
        "edge_meta":E[["veh_id","pos_id"]].reset_index(drop=True),
        "pos_meta":P[["pos_id","deck","lane"]].reset_index(drop=True),
    }

dataset={iid:make_instance(iid) for iid in inst.instance_id}
split_ids={s:splits.loc[splits.split==s,"instance_id"].tolist() for s in ["train","validation","test"]}

class BipartiteScorer(nn.Module):
    def __init__(self,v_in,p_in,e_in,h=32):
        super().__init__()
        self.venc=nn.Sequential(nn.Linear(v_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.penc=nn.Sequential(nn.Linear(p_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.vupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.pupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.score=nn.Sequential(
            nn.Linear(4*h+e_in,64),nn.ReLU(),nn.Dropout(0.05),
            nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)
        )
    def forward(self,d):
        vh=self.venc(d["vx"]); ph=self.penc(d["px"])
        vi=d["vi"]; pi=d["pi"]
        vm=torch.zeros_like(vh); vc=torch.zeros((len(vh),1),dtype=vh.dtype)
        pm=torch.zeros_like(ph); pc=torch.zeros((len(ph),1),dtype=ph.dtype)
        vm.index_add_(0,vi,ph[pi]); vc.index_add_(0,vi,torch.ones((len(vi),1)))
        pm.index_add_(0,pi,vh[vi]); pc.index_add_(0,pi,torch.ones((len(pi),1)))
        vm=vm/vc.clamp_min(1); pm=pm/pc.clamp_min(1)
        vh2=self.vupd(torch.cat([vh,vm],dim=1))
        ph2=self.pupd(torch.cat([ph,pm],dim=1))
        gv=vh2.mean(0); gp=ph2.mean(0)
        glob=torch.cat([gv,gp]).unsqueeze(0).expand(len(vi),-1)
        z=torch.cat([vh2[vi],ph2[pi],d["ex"],glob],dim=1)
        return self.score(z).squeeze(1)

model=BipartiteScorer(dataset[next(iter(dataset))]["vx"].shape[1],
                      dataset[next(iter(dataset))]["px"].shape[1],
                      dataset[next(iter(dataset))]["ex"].shape[1],h=32)

# Positive rate about 7%; use moderate positive weighting.
all_train_y=torch.cat([dataset[i]["y"] for i in split_ids["train"]])
pos_weight=torch.tensor((len(all_train_y)-all_train_y.sum())/all_train_y.sum())
criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5)

best_state=None; best_val=-1; patience=20; bad=0

def recall_k(ids,K=6):
    model.eval(); hits=0; total=0
    with torch.no_grad():
        for iid in ids:
            d=dataset[iid]; s=torch.sigmoid(model(d)).numpy()
            E=d["edge_meta"].copy(); E["score"]=s; E["label"]=d["y"].numpy()
            P=d["pos_meta"].set_index("pos_id")
            for vid,g in E.groupby("veh_id"):
                posg=g[g.label==1]
                if len(posg)!=1: continue
                target=int(posg.iloc[0].pos_id)
                top=set(g.nlargest(min(K,len(g)),"score").pos_id.astype(int))
                hits += int(target in top); total += 1
    return hits/total if total else np.nan

for epoch in range(1,101):
    model.train()
    random.shuffle(split_ids["train"])
    opt.zero_grad()
    losses=[]
    for iid in split_ids["train"]:
        d=dataset[iid]; logits=model(d); loss=criterion(logits,d["y"])
        loss.backward(); losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
    opt.step()
    if epoch%10==0:
        val=recall_k(split_ids["validation"],6)
        if val>best_val+1e-5:
            best_val=val; bad=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=10
        if bad>=patience: break

if best_state is not None: model.load_state_dict(best_state)

def evaluate(ids,ks=(1,2,3,4,6,8,12)):
    model.eval()
    rows=[]; predictions=[]
    with torch.no_grad():
        for iid in ids:
            d=dataset[iid]; scores=torch.sigmoid(model(d)).numpy()
            E=d["edge_meta"].copy()
            E["label"]=d["y"].numpy(); E["score"]=scores; E["instance_id"]=iid
            predictions.append(E)
    pred=pd.concat(predictions,ignore_index=True)
    pos_targets=pred[pred.label==1][["instance_id","veh_id","pos_id"]].rename(columns={"pos_id":"target_pos"})
    pos_lookup=pos.set_index(["instance_id","pos_id"])
    groups=pred.groupby(["instance_id","veh_id"])
    for K in ks:
        eh=dh=n=0
        for _,r in pos_targets.iterrows():
            g=groups.get_group((r.instance_id,r.veh_id)).sort_values("score",ascending=False)
            top=g.head(min(K,len(g)))
            target=int(r.target_pos)
            target_deck=int(pos_lookup.loc[(r.instance_id,target),"deck"])
            eh += int(target in set(top.pos_id.astype(int)))
            decks={int(pos_lookup.loc[(r.instance_id,int(pid)),"deck"]) for pid in top.pos_id}
            dh += int(target_deck in decks); n+=1
        rows.append({"K":K,"n_loaded_vehicles":n,
                     "exact_position_recall":eh/n,"deck_recall":dh/n})
    return pd.DataFrame(rows),pred

train_rec,_=evaluate(split_ids["train"])
val_rec,_=evaluate(split_ids["validation"])
test_rec,pred=evaluate(split_ids["test"])
train_rec.to_csv(OUT/"recall_train.csv",index=False)
val_rec.to_csv(OUT/"recall_validation.csv",index=False)
test_rec.to_csv(OUT/"recall_test.csv",index=False)
pred.to_csv(OUT/"test_edge_predictions.csv",index=False)
torch.save({"state_dict":model.state_dict(),"classes":classes,
            "vmu":vmu,"vsd":vsd,"pmu":pmu,"psd":psd,"emu":emu,"esd":esd},
           OUT/"bipartite_graph_scorer.pt")

metrics={"best_validation_recall_at_6":best_val,
         "epochs_run":epoch,
         "test_recall":test_rec.to_dict(orient="records")}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print("Epochs:",epoch,"best val R@6:",best_val)
print(test_rec.to_string(index=False))
