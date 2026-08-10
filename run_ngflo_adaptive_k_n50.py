
"""
Adaptive-K NGFLO evaluation on harder n=50 instances.

Policy:
- Start with K=6.
- Expand to K=8 if there is any deferral OR if more than 10% of loaded
  assignments use the lowest-ranked retained candidate ("boundary rate").
- Expand to K=12 under the same trigger.
- Stop otherwise.

The trigger is observable at deployment time and does not use the full-MIP optimum.

Evaluation compares the adaptive result with:
- a 5-second unrestricted Full-MIP incumbent on all regimes;
- certified full-MIP objective where available from prior n=50 certification
  (balanced and car-dense seed 33).

The graph scorer is trained only on n<=40, so n=50 is an out-of-distribution size test.
"""

from pathlib import Path
import sys, importlib.util, time, json, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import milp, Bounds

ROOT=Path("/mnt/data")
OUT=ROOT/"ngflo_adaptive_k_n50_v1"
OUT.mkdir(exist_ok=True)

# Load ferry model.
spec=importlib.util.spec_from_file_location("base_mod",ROOT/"synthetic_ferry_ordered_full_mip.py")
base=importlib.util.module_from_spec(spec); sys.modules["base_mod"]=base; spec.loader.exec_module(base)

# Load trained graph scorer checkpoint.
ckpt=torch.load(ROOT/"ngflo_graph_scorer_v2"/"bipartite_graph_scorer.pt",map_location="cpu",weights_only=False)
classes=ckpt["classes"]; class_to_idx={c:i for i,c in enumerate(classes)}
vmu=np.asarray(ckpt["vmu"],dtype=np.float32); vsd=np.asarray(ckpt["vsd"],dtype=np.float32)
pmu=np.asarray(ckpt["pmu"],dtype=np.float32); psd=np.asarray(ckpt["psd"],dtype=np.float32)
emu=np.asarray(ckpt["emu"],dtype=np.float32); esd=np.asarray(ckpt["esd"],dtype=np.float32)

veh_num=["length","width","height","mass","destination","priority"]
pos_num=["deck","lane","length_cap","width_cap","height_cap","mass_cap","y_coord"]
edge_num=["length_ratio","mass_ratio","abs_y_mass","destination_deck_distance"]

class BipartiteScorer(nn.Module):
    def __init__(self,v_in,p_in,e_in,h=32):
        super().__init__()
        self.venc=nn.Sequential(nn.Linear(v_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.penc=nn.Sequential(nn.Linear(p_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.vupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.pupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.score=nn.Sequential(nn.Linear(4*h+e_in,64),nn.ReLU(),nn.Dropout(0.05),
                                 nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,d):
        vh=self.venc(d["vx"]); ph=self.penc(d["px"]); vi=d["vi"]; pi=d["pi"]
        vm=torch.zeros_like(vh); vc=torch.zeros((len(vh),1))
        pm=torch.zeros_like(ph); pc=torch.zeros((len(ph),1))
        ones=torch.ones((len(vi),1))
        vm.index_add_(0,vi,ph[pi]); vc.index_add_(0,vi,ones)
        pm.index_add_(0,pi,vh[vi]); pc.index_add_(0,pi,ones)
        vm=vm/vc.clamp_min(1); pm=pm/pc.clamp_min(1)
        vh2=self.vupd(torch.cat([vh,vm],1)); ph2=self.pupd(torch.cat([ph,pm],1))
        glob=torch.cat([vh2.mean(0),ph2.mean(0)]).unsqueeze(0).expand(len(vi),-1)
        z=torch.cat([vh2[vi],ph2[pi],d["ex"],glob],1)
        return self.score(z).squeeze(1)

model=BipartiteScorer(len(vmu)+len(classes),len(pmu),len(emu),32)
model.load_state_dict(ckpt["state_dict"]); model.eval(); torch.set_num_threads(1)

def score_instance(vehicles,positions,cfg):
    V=vehicles.sort_values("veh_id").copy(); P=positions.sort_values("pos_id").copy()
    vmap={int(x):i for i,x in enumerate(V.veh_id)}
    pmap={int(x):i for i,x in enumerate(P.pos_id)}
    vn=(V[veh_num].to_numpy(np.float32)-vmu)/vsd
    one=np.zeros((len(V),len(classes)),np.float32)
    for q,c in enumerate(V["class"]): one[q,class_to_idx[c]]=1
    vx=np.concatenate([vn,one],1)
    px=(P[pos_num].to_numpy(np.float32)-pmu)/psd

    erows=[]
    for _,v in V.iterrows():
        for _,p in P.iterrows():
            if base.compatible(v,p,cfg):
                erows.append({
                    "veh_id":int(v.veh_id),"pos_id":int(p.pos_id),
                    "length_ratio":float(v.length/p.length_cap),
                    "mass_ratio":float(v.mass/p.mass_cap),
                    "abs_y_mass":float(abs(p.y_coord)*v.mass),
                    "destination_deck_distance":float(abs((int(v.destination)-1)-min(int(p.deck),2)))
                })
    E=pd.DataFrame(erows)
    ex=(E[edge_num].to_numpy(np.float32)-emu)/esd
    vi=np.array([vmap[int(x)] for x in E.veh_id],np.int64)
    pi=np.array([pmap[int(x)] for x in E.pos_id],np.int64)
    d={"vx":torch.tensor(vx),"px":torch.tensor(px),"ex":torch.tensor(ex),
       "vi":torch.tensor(vi),"pi":torch.tensor(pi)}
    with torch.no_grad():
        E["score"]=torch.sigmoid(model(d)).numpy()
    E["rank"]=E.groupby("veh_id")["score"].rank(method="first",ascending=False).astype(int)
    return E

def solve_restricted(vehicles,positions,cfg,E,K,time_limit=2.0):
    c,integrality,bounds,constraints,m=base.build_ordered_slot_model(vehicles,positions,cfg)
    allowed={}
    for vid,g in E.groupby("veh_id"):
        allowed[int(vid)]=set(g.nlargest(min(K,len(g)),"score").pos_id.astype(int))
    lb=np.zeros(len(c)); ub=np.ones(len(c)); retained=0
    for idx,(i,p,k) in enumerate(m["x_vars"]):
        if p not in allowed.get(int(i),set()): ub[idx]=0
        else: retained+=1
    t0=time.perf_counter()
    res=milp(c=c,integrality=integrality,bounds=Bounds(lb,ub),constraints=constraints,
             options={"time_limit":time_limit,"mip_rel_gap":1e-5,"presolve":True})
    rt=time.perf_counter()-t0
    if res.x is None:
        return {"success":False,"K":K,"runtime_s":rt,"retained":retained,"total_x":len(m["x_vars"])}
    x=np.asarray(res.x); loaded=set(); selected=[]
    rank_lookup=E.set_index(["veh_id","pos_id"])["rank"].to_dict()
    for idx,(i,p,k) in enumerate(m["x_vars"]):
        if x[idx]>0.5:
            loaded.add(int(i)); selected.append((int(i),int(p),int(rank_lookup[(int(i),int(p))])))
    boundary=sum(r>=K for _,_,r in selected)/max(1,len(selected))
    return {
        "success":True,"K":K,"runtime_s":rt,"objective":float(res.fun),
        "mip_gap":float(getattr(res,"mip_gap",np.nan)),
        "loaded":len(loaded),"deferred":len(vehicles)-len(loaded),
        "boundary_rate":boundary,
        "retained":retained,"total_x":len(m["x_vars"]),
        "variable_retention":retained/len(m["x_vars"]),
    }

# Certified objectives available from prior files.
cert={}
for p in [ROOT/"cert_balanced_n50.csv",ROOT/"cert_car_dense_n50.csv"]:
    if p.exists():
        d=pd.read_csv(p)
        for _,r in d.iterrows():
            cert[(str(r.congestion),int(r.seed))]=float(r.objective)

rows=[]; stages=[]
for congestion in ["balanced","heavy","car_dense"]:
    seed=33; n=50; cfg=base.FERRY_PRESETS["small"]
    vehicles=base.generate_vehicles(n,seed,congestion); positions=base.make_positions(cfg)
    E=score_instance(vehicles,positions,cfg)

    # Full MIP incumbent under same 5-second budget.
    c,integ,bounds,cons,m=base.build_ordered_slot_model(vehicles,positions,cfg)
    t0=time.perf_counter()
    full=milp(c=c,integrality=integ,bounds=bounds,constraints=cons,
              options={"time_limit":5.0,"mip_rel_gap":1e-6,"presolve":True})
    full_rt=time.perf_counter()-t0
    full_obj=float(full.fun) if full.x is not None else np.nan
    full_gap=float(getattr(full,"mip_gap",np.nan))

    total_adapt_time=0.0; final=None
    for K in [6,8,12]:
        r=solve_restricted(vehicles,positions,cfg,E,K,2.0)
        total_adapt_time += r["runtime_s"]
        r.update({"congestion":congestion,"seed":seed,"n":n})
        stages.append(r.copy())
        final=r
        trigger=(not r["success"]) or r.get("deferred",1)>0 or r.get("boundary_rate",1.0)>0.10
        if not trigger:
            break

    cert_obj=cert.get((congestion,seed),np.nan)
    rows.append({
        "congestion":congestion,"n":n,"seed":seed,
        "final_K":final["K"],"adaptive_runtime_s":total_adapt_time,
        "adaptive_objective":final.get("objective",np.nan),
        "adaptive_loaded":final.get("loaded",np.nan),
        "adaptive_deferred":final.get("deferred",np.nan),
        "adaptive_boundary_rate":final.get("boundary_rate",np.nan),
        "adaptive_variable_retention":final.get("variable_retention",np.nan),
        "full_5s_runtime_s":full_rt,"full_5s_objective":full_obj,"full_5s_mip_gap":full_gap,
        "certified_objective":cert_obj,
        "gap_vs_certified_pct":(
            100*(final["objective"]-cert_obj)/abs(cert_obj)
            if np.isfinite(cert_obj) and final.get("objective") is not None else np.nan
        ),
        "gap_vs_full5s_incumbent_pct":(
            100*(final["objective"]-full_obj)/abs(full_obj)
            if np.isfinite(full_obj) and final.get("objective") is not None else np.nan
        ),
        "speedup_vs_full5s":full_rt/total_adapt_time if total_adapt_time>0 else np.nan,
    })
    print(rows[-1])

pd.DataFrame(stages).to_csv(OUT/"adaptive_stage_results.csv",index=False)
final_df=pd.DataFrame(rows)
final_df.to_csv(OUT/"adaptive_n50_results.csv",index=False)
print("\nFINAL")
print(final_df.to_string(index=False))
