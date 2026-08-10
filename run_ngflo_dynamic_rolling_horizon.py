
"""
Dynamic stochastic vehicle-ferry loading environment with rolling-horizon control.

This stage extends the static NGFLO prototype to multiple decision epochs with:
- stochastic arrivals;
- optional cancellations;
- carry-over waiting vehicles;
- repeated re-optimization;
- dynamic performance accounting.

Controllers:
1. RH-Full-MIP: unrestricted ordered-slot MIP at every decision epoch.
2. RH-NGFLO: hardness-routed adaptive NGFLO with K=6 -> 8 -> 12 and full-search fallback.

Important simplification
------------------------
This first dynamic prototype models each epoch as a *planning state* before final vessel departure.
Vehicles not selected remain in the waiting pool and may be reconsidered later. We do not yet freeze
physically loaded vehicle positions across epochs. A commitment/frozen-assignment model will be the
next refinement once the stochastic pipeline is validated.

The code reuses the synthetic small-ferry geometry and the trained static hardness/scoring models.
"""

from pathlib import Path
import sys, importlib.util, time, math, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import milp, Bounds, LinearConstraint
import joblib

ROOT = Path("/mnt/data")
OUT = ROOT / "ngflo_dynamic_rh_v1"
OUT.mkdir(exist_ok=True)

# ---------- Load base ferry model ----------
spec = importlib.util.spec_from_file_location(
    "base_mod", ROOT / "synthetic_ferry_ordered_full_mip.py"
)
base = importlib.util.module_from_spec(spec)
sys.modules["base_mod"] = base
spec.loader.exec_module(base)

# ---------- Hardness router ----------
router = joblib.load(ROOT/"ngflo_hardness_router_v1"/"hardness_router.joblib")
ROUTER_THRESHOLD = 0.40
router_num = [
    "n","total_vehicle_length","mean_vehicle_length","std_vehicle_length",
    "max_vehicle_length","total_vehicle_mass","mean_vehicle_mass",
    "std_vehicle_mass","max_vehicle_mass","mean_vehicle_height","max_vehicle_height",
    "length_pressure","mass_pressure","mean_compatible_positions",
    "min_compatible_positions","mean_compatible_decks","min_compatible_decks",
    "compatibility_density","p_car","p_suv","p_van","p_rigid_truck",
    "p_coach","p_artic","p_heavy","destination_entropy"
]
router_cat = ["congestion"]

# ---------- Graph scorer ----------
ckpt = torch.load(
    ROOT/"ngflo_graph_scorer_v2"/"bipartite_graph_scorer.pt",
    map_location="cpu", weights_only=False
)
classes = ckpt["classes"]
class_to_idx = {c:i for i,c in enumerate(classes)}
vmu = np.asarray(ckpt["vmu"], dtype=np.float32)
vsd = np.asarray(ckpt["vsd"], dtype=np.float32)
pmu = np.asarray(ckpt["pmu"], dtype=np.float32)
psd = np.asarray(ckpt["psd"], dtype=np.float32)
emu = np.asarray(ckpt["emu"], dtype=np.float32)
esd = np.asarray(ckpt["esd"], dtype=np.float32)

veh_num = ["length","width","height","mass","destination","priority"]
pos_num = ["deck","lane","length_cap","width_cap","height_cap","mass_cap","y_coord"]
edge_num = ["length_ratio","mass_ratio","abs_y_mass","destination_deck_distance"]

class BipartiteScorer(nn.Module):
    def __init__(self, v_in, p_in, e_in, h=32):
        super().__init__()
        self.venc = nn.Sequential(nn.Linear(v_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.penc = nn.Sequential(nn.Linear(p_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.vupd = nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.pupd = nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.score = nn.Sequential(
            nn.Linear(4*h+e_in,64),nn.ReLU(),nn.Dropout(0.05),
            nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)
        )
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

scorer = BipartiteScorer(len(vmu)+len(classes), len(pmu), len(emu), 32)
scorer.load_state_dict(ckpt["state_dict"])
scorer.eval()
torch.set_num_threads(1)

# ---------- stochastic environment ----------
def arrival_count(rng, regime, epoch, horizon):
    if regime == "poisson":
        lam = 8.0
    elif regime == "nonhomogeneous":
        # build toward departure
        lam = 4.0 + 8.0 * epoch / max(1, horizon-1)
    elif regime == "bursty":
        # simple two-state Markov-modulated proxy
        lam = 4.0 if rng.random() < 0.65 else 15.0
    else:
        raise ValueError(regime)
    return int(rng.poisson(lam))

def generate_arrivals(n, seed, congestion, start_id):
    if n <= 0:
        return pd.DataFrame(columns=[
            "veh_id","class","length","width","height","mass","destination","priority"
        ])
    df = base.generate_vehicles(n, seed, congestion).copy()
    df["veh_id"] = np.arange(start_id, start_id+n)
    return df

def hardness_probability(vehicles,positions,cfg,congestion):
    if len(vehicles)==0:
        return 0.0
    compat_counts=[]; compat_decks=[]
    for _,v in vehicles.iterrows():
        poss=[]; decks=set()
        for _,p in positions.iterrows():
            if base.compatible(v,p,cfg):
                poss.append(int(p.pos_id)); decks.add(int(p.deck))
        compat_counts.append(len(poss)); compat_decks.append(len(decks))
    cc=vehicles["class"].value_counts(normalize=True).to_dict()
    probs=vehicles.destination.value_counts(normalize=True).values
    tl=float(positions.length_cap.sum()); tm=float(positions.mass_cap.sum())
    row={
        "n":len(vehicles),
        "total_vehicle_length":float(vehicles.length.sum()),
        "mean_vehicle_length":float(vehicles.length.mean()),
        "std_vehicle_length":float(vehicles.length.std(ddof=0)),
        "max_vehicle_length":float(vehicles.length.max()),
        "total_vehicle_mass":float(vehicles.mass.sum()),
        "mean_vehicle_mass":float(vehicles.mass.mean()),
        "std_vehicle_mass":float(vehicles.mass.std(ddof=0)),
        "max_vehicle_mass":float(vehicles.mass.max()),
        "mean_vehicle_height":float(vehicles.height.mean()),
        "max_vehicle_height":float(vehicles.height.max()),
        "length_pressure":float(vehicles.length.sum()/tl),
        "mass_pressure":float(vehicles.mass.sum()/tm),
        "mean_compatible_positions":float(np.mean(compat_counts)),
        "min_compatible_positions":int(np.min(compat_counts)),
        "mean_compatible_decks":float(np.mean(compat_decks)),
        "min_compatible_decks":int(np.min(compat_decks)),
        "compatibility_density":float(np.sum(compat_counts)/(len(vehicles)*len(positions))),
        "p_car":float(cc.get("car",0)),"p_suv":float(cc.get("suv",0)),
        "p_van":float(cc.get("van",0)),"p_rigid_truck":float(cc.get("rigid_truck",0)),
        "p_coach":float(cc.get("coach",0)),"p_artic":float(cc.get("artic",0)),
        "p_heavy":float(cc.get("rigid_truck",0)+cc.get("coach",0)+cc.get("artic",0)),
        "destination_entropy":float(-sum(p*math.log(max(p,1e-12)) for p in probs)),
        "congestion":congestion
    }
    X=pd.DataFrame([row])
    return float(router.predict_proba(X[router_num+router_cat])[:,1][0])

def score_edges(vehicles,positions,cfg):
    V=vehicles.sort_values("veh_id").copy()
    P=positions.sort_values("pos_id").copy()
    vmap={int(x):i for i,x in enumerate(V.veh_id)}
    pmap={int(x):i for i,x in enumerate(P.pos_id)}
    vn=(V[veh_num].to_numpy(np.float32)-vmu)/vsd
    one=np.zeros((len(V),len(classes)),np.float32)
    for q,c in enumerate(V["class"]):
        one[q,class_to_idx[c]]=1
    vx=np.concatenate([vn,one],1)
    px=(P[pos_num].to_numpy(np.float32)-pmu)/psd

    rows=[]
    for _,v in V.iterrows():
        for _,p in P.iterrows():
            if base.compatible(v,p,cfg):
                rows.append({
                    "veh_id":int(v.veh_id),"pos_id":int(p.pos_id),
                    "length_ratio":float(v.length/p.length_cap),
                    "mass_ratio":float(v.mass/p.mass_cap),
                    "abs_y_mass":float(abs(p.y_coord)*v.mass),
                    "destination_deck_distance":float(
                        abs((int(v.destination)-1)-min(int(p.deck),2))
                    )
                })
    E=pd.DataFrame(rows)
    ex=(E[edge_num].to_numpy(np.float32)-emu)/esd
    vi=np.array([vmap[int(x)] for x in E.veh_id],np.int64)
    pi=np.array([pmap[int(x)] for x in E.pos_id],np.int64)
    d={"vx":torch.tensor(vx),"px":torch.tensor(px),"ex":torch.tensor(ex),
       "vi":torch.tensor(vi),"pi":torch.tensor(pi)}
    with torch.no_grad():
        E["score"]=torch.sigmoid(scorer(d)).numpy()
    E["rank"]=E.groupby("veh_id")["score"].rank(method="first",ascending=False).astype(int)
    return E

def solve_state(vehicles,positions,cfg,time_limit=0.75,allowed=None,cutoff=None):
    c,integ,bounds,cons,m=base.build_ordered_slot_model(vehicles,positions,cfg)
    lb=np.zeros(len(c)); ub=np.ones(len(c))
    if allowed is not None:
        for idx,(i,p,k) in enumerate(m["x_vars"]):
            if p not in allowed.get(int(i),set()):
                ub[idx]=0.0
    constraints=[cons]
    if cutoff is not None and np.isfinite(cutoff):
        constraints.append(LinearConstraint(c,-np.inf,float(cutoff)+1e-9))
    t0=time.perf_counter()
    res=milp(c=c,integrality=integ,bounds=Bounds(lb,ub),constraints=constraints,
             options={"time_limit":time_limit,"mip_rel_gap":1e-5,"presolve":True})
    rt=time.perf_counter()-t0
    if res.x is None:
        return {"success":False,"runtime_s":rt,"objective":np.nan,"loaded_ids":set(),
                "deferred":len(vehicles),"mip_gap":np.nan,"solution_x":None,"meta":m}
    x=np.asarray(res.x); loaded=set()
    for idx,(i,p,k) in enumerate(m["x_vars"]):
        if x[idx]>0.5: loaded.add(int(i))
    return {
        "success":True,"runtime_s":rt,"objective":float(res.fun),"loaded_ids":loaded,
        "deferred":len(vehicles)-len(loaded),
        "mip_gap":float(getattr(res,"mip_gap",np.nan)),"solution_x":x,"meta":m
    }

def adaptive_ngflo(vehicles,positions,cfg):
    E=score_edges(vehicles,positions,cfg)
    total=0.0; final=None; stage_count=0
    for K in [6,8,12]:
        allowed={int(vid):set(g.nlargest(min(K,len(g)),"score").pos_id.astype(int))
                 for vid,g in E.groupby("veh_id")}
        r=solve_state(vehicles,positions,cfg,time_limit=0.5,allowed=allowed)
        total+=r["runtime_s"]; stage_count+=1; final=r
        if r["success"]:
            rank_lookup=E.set_index(["veh_id","pos_id"])["rank"].to_dict()
            ranks=[]
            for idx,(i,p,k) in enumerate(r["meta"]["x_vars"]):
                if r["solution_x"][idx]>0.5:
                    ranks.append(int(rank_lookup[(int(i),int(p))]))
            boundary=sum(rr>=K for rr in ranks)/max(1,len(ranks))
        else:
            boundary=1.0
        final["K"]=K; final["boundary_rate"]=boundary
        if r["success"] and r["deferred"]==0 and boundary<=0.10:
            break

    fallback=False; fallback_rt=0.0
    if (not final["success"]) or final["deferred"]>0 or final["boundary_rate"]>0.10:
        fallback=True
        incumbent=final["objective"] if final["success"] else None
        fb=solve_state(vehicles,positions,cfg,time_limit=1.0,allowed=None,cutoff=incumbent)
        fallback_rt=fb["runtime_s"]
        if fb["success"] and ((not final["success"]) or fb["objective"]<=final["objective"]+1e-9):
            final=fb
            final["K"]=len(positions)
            final["boundary_rate"]=0.0
    return final,total+fallback_rt,stage_count,fallback

def controller_full(vehicles,positions,cfg):
    r=solve_state(vehicles,positions,cfg,time_limit=0.75)
    return r,r["runtime_s"],1,False,"Full-MIP",np.nan

def controller_ngflo(vehicles,positions,cfg,congestion):
    ph=hardness_probability(vehicles,positions,cfg,congestion)
    if ph<ROUTER_THRESHOLD:
        r=solve_state(vehicles,positions,cfg,time_limit=0.75)
        return r,r["runtime_s"],1,False,"Full-MIP",ph
    r,rt,stages,fb=adaptive_ngflo(vehicles,positions,cfg)
    return r,rt,stages,fb,"Adaptive-NGFLO",ph

def run_episode(controller,arrival_regime,congestion,seed,horizon=6,cancel_prob=0.03):
    rng=np.random.default_rng(seed)
    cfg=base.FERRY_PRESETS["small"]
    positions=base.make_positions(cfg)
    waiting=pd.DataFrame(columns=[
        "veh_id","class","length","width","height","mass","destination","priority","arrival_epoch"
    ])
    next_id=0
    prev_loaded=set()
    logs=[]
    cumulative_wait=0.0
    total_arrivals=0
    total_cancellations=0
    total_runtime=0.0

    for t in range(horizon):
        # cancellations from waiting pool before new arrivals
        if len(waiting):
            cancel_mask=rng.random(len(waiting))<cancel_prob
            cancelled_ids=set(waiting.loc[cancel_mask,"veh_id"].astype(int))
            total_cancellations+=len(cancelled_ids)
            waiting=waiting.loc[~cancel_mask].reset_index(drop=True)

        na=arrival_count(rng,arrival_regime,t,horizon)
        arrivals=generate_arrivals(na,seed*100+t,congestion,next_id)
        next_id+=na; total_arrivals+=na
        if len(arrivals):
            arrivals["arrival_epoch"]=t
            waiting=pd.concat([waiting,arrivals],ignore_index=True)

        if len(waiting)==0:
            logs.append({"epoch":t,"arrivals":na,"waiting":0,"loaded":0,"deferred":0,
                         "runtime_s":0.0,"objective":0.0,"rehandles":0,
                         "fallback":False,"route":"none","hard_probability":0.0})
            continue

        if controller=="ngflo":
            r,rt,stages,fb,route,ph=controller_ngflo(waiting,positions,cfg,congestion)
        else:
            r,rt,stages,fb,route,ph=controller_full(waiting,positions,cfg)

        total_runtime+=rt
        loaded=set(r["loaded_ids"])
        deferred=set(waiting.veh_id.astype(int))-loaded
        rehandles=len(prev_loaded.symmetric_difference(loaded))
        prev_loaded=loaded

        # waiting cost: deferred vehicle-age sum
        ages=waiting.set_index("veh_id")["arrival_epoch"].to_dict()
        epoch_wait=sum(t-int(ages[i])+1 for i in deferred)
        cumulative_wait+=epoch_wait

        logs.append({
            "epoch":t,"arrivals":na,"waiting":len(waiting),"loaded":len(loaded),
            "deferred":len(deferred),"runtime_s":rt,"objective":r["objective"],
            "mip_gap":r["mip_gap"],"rehandles":rehandles,"fallback":fb,
            "route":route,"hard_probability":ph,"stages":stages,
            "mean_wait_age":float(np.mean([t-int(ages[i])+1 for i in deferred])) if deferred else 0.0
        })

    final_loaded=prev_loaded
    final_deferred=set(waiting.veh_id.astype(int))-final_loaded
    # Dynamic score combines final solver objective, cumulative waiting, and plan churn.
    churn=sum(x["rehandles"] for x in logs)
    final_obj=logs[-1]["objective"] if logs else 0.0
    dynamic_score=float(final_obj + 2.0*cumulative_wait + 0.15*churn)

    summary={
        "controller":controller,"arrival_regime":arrival_regime,"congestion":congestion,
        "seed":seed,"horizon":horizon,"total_arrivals":total_arrivals,
        "total_cancellations":total_cancellations,"final_waiting_pool":len(waiting),
        "final_loaded":len(final_loaded),"final_deferred":len(final_deferred),
        "cumulative_wait_units":cumulative_wait,"total_rehandles":churn,
        "total_runtime_s":total_runtime,"fallback_count":sum(bool(x["fallback"]) for x in logs),
        "dynamic_score":dynamic_score
    }
    return summary,logs

summaries=[]; logs=[]
for arrival_regime in ["poisson","nonhomogeneous","bursty"]:
    for congestion in ["balanced","heavy"]:
        for seed in [101]:
            for controller in ["full","ngflo"]:
                s,l=run_episode(controller,arrival_regime,congestion,seed,horizon=4)
                summaries.append(s)
                for row in l:
                    row.update({
                        "controller":controller,"arrival_regime":arrival_regime,
                        "congestion":congestion,"seed":seed
                    })
                    logs.append(row)
                print(s)

sdf=pd.DataFrame(summaries)
ldf=pd.DataFrame(logs)
sdf.to_csv(OUT/"dynamic_episode_summary.csv",index=False)
ldf.to_csv(OUT/"dynamic_epoch_log.csv",index=False)

comp=(
    sdf.pivot_table(
        index=["arrival_regime","congestion","seed"],
        columns="controller",
        values=["dynamic_score","total_runtime_s","final_loaded","final_deferred",
                "cumulative_wait_units","total_rehandles","fallback_count"]
    )
)
comp.columns=["_".join(c) for c in comp.columns]
comp=comp.reset_index()
comp["runtime_ratio_ngflo_full"]=comp["total_runtime_s_ngflo"]/comp["total_runtime_s_full"]
comp["score_change_pct"]=100*(comp["dynamic_score_ngflo"]-comp["dynamic_score_full"])/abs(comp["dynamic_score_full"])
comp.to_csv(OUT/"dynamic_controller_comparison.csv",index=False)

print("\nAGGREGATE")
agg=sdf.groupby(["controller","arrival_regime","congestion"]).agg(
    runtime_mean=("total_runtime_s","mean"),
    score_mean=("dynamic_score","mean"),
    final_loaded_mean=("final_loaded","mean"),
    final_deferred_mean=("final_deferred","mean"),
    wait_mean=("cumulative_wait_units","mean"),
    rehandles_mean=("total_rehandles","mean"),
    fallback_mean=("fallback_count","mean"),
).reset_index()
agg.to_csv(OUT/"dynamic_aggregate_summary.csv",index=False)
print(agg.to_string(index=False))
