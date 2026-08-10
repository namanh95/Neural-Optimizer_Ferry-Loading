
"""
Commitment-aware rolling-horizon NGFLO environment.

Key refinement over v1:
- A committed vehicle keeps its lane/deck/slot assignment across future epochs.
- Only uncommitted/newly arrived vehicles are re-optimized.
- Residual lane length, lane mass, deck mass, and transverse moment capacity
  are recomputed from the committed set before each solve.
- Cancellations affect only uncommitted vehicles.
- Commitment occurs after each optimization according to a configurable policy.

Controllers:
1. Commitment-aware RH-Full-MIP.
2. Commitment-aware RH-NGFLO with hardness routing, adaptive K, and full-search fallback.

This is still a synthetic benchmark. It is not an operator-specific stowage model.
"""

from pathlib import Path
import sys, importlib.util, time, math, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import vstack
import joblib

ROOT = Path("/mnt/data")
OUT = ROOT / "ngflo_publication_pilot_v1"
OUT.mkdir(exist_ok=True)

# ---------- Load baseline ----------
spec = importlib.util.spec_from_file_location(
    "base_mod", ROOT / "synthetic_ferry_ordered_full_mip.py"
)
base = importlib.util.module_from_spec(spec)
sys.modules["base_mod"] = base
spec.loader.exec_module(base)

# ---------- Router ----------
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

scorer = BipartiteScorer(len(vmu)+len(classes),len(pmu),len(emu),32)
scorer.load_state_dict(ckpt["state_dict"])
scorer.eval()
torch.set_num_threads(1)

# ---------- arrivals ----------
ARRIVAL_SCALE = 1.0

def arrival_count(rng, regime, epoch, horizon):
    if regime == "poisson":
        lam = 8.0
    elif regime == "nonhomogeneous":
        lam = 4.0 + 8.0 * epoch / max(1,horizon-1)
    elif regime == "bursty":
        lam = 4.0 if rng.random() < 0.65 else 15.0
    else:
        raise ValueError(regime)
    return int(rng.poisson(ARRIVAL_SCALE * lam))

def generate_arrivals(n, seed, congestion, start_id):
    if n <= 0:
        return pd.DataFrame(columns=[
            "veh_id","class","length","width","height","mass","destination","priority"
        ])
    df=base.generate_vehicles(n,seed,congestion).copy()
    df["veh_id"]=np.arange(start_id,start_id+n)
    return df

# ---------- committed state ----------
def committed_usage(committed, positions):
    """
    committed columns:
      veh_id,pos_id,slot,length,mass,deck,lane,destination,priority,class,height,width
    """
    lane_length = {int(p):0.0 for p in positions.pos_id}
    lane_mass = {int(p):0.0 for p in positions.pos_id}
    deck_mass = {int(d):0.0 for d in positions.deck.unique()}
    trans_moment = 0.0
    occupied_slots = set()

    pidx=positions.set_index("pos_id")
    for _,r in committed.iterrows():
        pid=int(r.pos_id)
        lane_length[pid]+=float(r.length)
        lane_mass[pid]+=float(r.mass)
        deck_mass[int(r.deck)]+=float(r.mass)
        trans_moment += float(r.mass)*float(pidx.loc[pid,"y_coord"])
        occupied_slots.add((pid,int(r.slot)))
    return lane_length,lane_mass,deck_mass,trans_moment,occupied_slots

def build_residual_model(uncommitted, positions, cfg, committed):
    """
    Build the ordinary ordered-slot model on uncommitted vehicles, then tighten
    capacities and slot availability using committed usage.
    """
    c, integ, bounds, cons, meta = base.build_ordered_slot_model(
        uncommitted, positions, cfg
    )

    lane_len_used,lane_mass_used,deck_mass_used,trans_used,occupied = committed_usage(
        committed, positions
    )

    # Residual slots are indexed locally for the uncommitted suffix of each lane.
    # Committed vehicles form a fixed prefix, so local residual slot 0 means the
    # first available position behind that prefix. No residual variable is fixed
    # merely because an absolute committed slot has the same integer label.
    lbv=np.zeros(len(c)); ubv=np.ones(len(c))

    # Add residual-capacity inequalities explicitly.
    veh=uncommitted.set_index("veh_id")
    pos=positions.set_index("pos_id")
    rows=[]; lbs=[]; ubs=[]

    # Residual lane length and mass.
    for pid in positions.pos_id.astype(int):
        row_len=np.zeros(len(c)); row_mass=np.zeros(len(c))
        for (i,p,k),idx in meta["x_idx"].items():
            if int(p)==pid:
                row_len[idx]=float(veh.loc[i,"length"])
                row_mass[idx]=float(veh.loc[i,"mass"])
        rows.append(row_len); lbs.append(-np.inf)
        ubs.append(float(pos.loc[pid,"length_cap"])-lane_len_used[pid])
        rows.append(row_mass); lbs.append(-np.inf)
        ubs.append(float(pos.loc[pid,"mass_cap"])-lane_mass_used[pid])

    # Residual deck mass.
    for d in range(cfg.decks):
        row=np.zeros(len(c))
        pset=set(positions.loc[positions.deck==d,"pos_id"].astype(int))
        for (i,p,k),idx in meta["x_idx"].items():
            if int(p) in pset:
                row[idx]=float(veh.loc[i,"mass"])
        rows.append(row); lbs.append(-np.inf)
        ubs.append(cfg.deck_mass_capacity-deck_mass_used[d])

    # Discharge-order bridge from committed prefix to residual suffix.
    # For each lane with committed vehicles, the first residual vehicle must
    # have destination rank >= the deepest committed destination rank.
    for pid in positions.pos_id.astype(int):
        c_lane=committed[committed.pos_id.astype(int)==pid] if len(committed) else committed
        if len(c_lane):
            last_dest=int(c_lane.sort_values("slot").iloc[-1].destination)
            # Forbid residual assignments in this lane whose destination is earlier.
            for (i,p,k),idx in meta["x_idx"].items():
                if int(p)==pid and int(veh.loc[i,"destination"]) < last_dest:
                    ubv[idx]=0.0

    # Residual transverse balance:
    # |M_comm + sum m*y*x| <= ratio*(mass_comm + sum m*x)
    pidx=positions.set_index("pos_id")
    committed_mass=float(committed.mass.sum()) if len(committed) else 0.0
    ratio=cfg.transverse_balance_ratio
    for sign in (1.0,-1.0):
        row=np.zeros(len(c))
        for (i,p,k),idx in meta["x_idx"].items():
            m=float(veh.loc[i,"mass"]); y=float(pidx.loc[p,"y_coord"])
            row[idx]=m*(sign*y-ratio)
        rhs=ratio*committed_mass - sign*trans_used
        rows.append(row); lbs.append(-np.inf); ubs.append(rhs)

    if rows:
        extra=LinearConstraint(np.vstack(rows),np.array(lbs),np.array(ubs))
        constraints=[cons,extra]
    else:
        constraints=[cons]

    return c,integ,Bounds(lbv,ubv),constraints,meta

def hardness_probability(vehicles,positions,cfg,congestion):
    if len(vehicles)==0: return 0.0
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
        "n":len(vehicles),"total_vehicle_length":float(vehicles.length.sum()),
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
        E["score"]=torch.sigmoid(scorer(d)).numpy()
    E["rank"]=E.groupby("veh_id")["score"].rank(method="first",ascending=False).astype(int)
    return E

def solve_residual(uncommitted,positions,cfg,committed,time_limit=0.40,allowed=None,cutoff=None):
    if len(uncommitted)==0:
        return {
            "success":True,"runtime_s":0.0,"objective":0.0,"loaded_ids":set(),
            "deferred":0,"mip_gap":0.0,"assignments":pd.DataFrame()
        }

    c,integ,bounds,constraints,meta=build_residual_model(
        uncommitted,positions,cfg,committed
    )
    lb=bounds.lb.copy(); ub=bounds.ub.copy()

    if allowed is not None:
        for idx,(i,p,k) in enumerate(meta["x_vars"]):
            if int(p) not in allowed.get(int(i),set()):
                ub[idx]=0.0

    cons=list(constraints)
    if cutoff is not None and np.isfinite(cutoff):
        cons.append(LinearConstraint(c,-np.inf,float(cutoff)+1e-9))

    t0=time.perf_counter()
    res=milp(
        c=c,integrality=integ,bounds=Bounds(lb,ub),constraints=cons,
        options={"time_limit":time_limit,"mip_rel_gap":1e-5,"presolve":True}
    )
    rt=time.perf_counter()-t0

    if res.x is None:
        return {
            "success":False,"runtime_s":rt,"objective":np.nan,"loaded_ids":set(),
            "deferred":len(uncommitted),"mip_gap":np.nan,"assignments":pd.DataFrame(),
            "solution_x":None,"meta":meta
        }

    x=np.asarray(res.x)
    veh=uncommitted.set_index("veh_id")
    pos=positions.set_index("pos_id")
    loaded=set(); arows=[]
    for idx,(i,p,k) in enumerate(meta["x_vars"]):
        if x[idx]>0.5:
            loaded.add(int(i))
            arows.append({
                "veh_id":int(i),"pos_id":int(p),"slot":int(k),
                "deck":int(pos.loc[p,"deck"]),"lane":int(pos.loc[p,"lane"]),
                "length":float(veh.loc[i,"length"]),"width":float(veh.loc[i,"width"]),
                "height":float(veh.loc[i,"height"]),"mass":float(veh.loc[i,"mass"]),
                "destination":int(veh.loc[i,"destination"]),
                "priority":int(veh.loc[i,"priority"]),
                "class":str(veh.loc[i,"class"])
            })

    return {
        "success":True,"runtime_s":rt,"objective":float(res.fun),
        "loaded_ids":loaded,"deferred":len(uncommitted)-len(loaded),
        "mip_gap":float(getattr(res,"mip_gap",np.nan)),
        "assignments":pd.DataFrame(arows),
        "solution_x":x,"meta":meta
    }

def adaptive_controller(uncommitted,positions,cfg,committed,congestion):
    ph=hardness_probability(uncommitted,positions,cfg,congestion)
    if ph<ROUTER_THRESHOLD:
        r=solve_residual(uncommitted,positions,cfg,committed,time_limit=0.40)
        return r,r["runtime_s"],1,False,"Full-MIP",ph

    E=score_edges(uncommitted,positions,cfg)
    total=0.0; final=None; stages=0
    for K in [6,8,12]:
        allowed={int(vid):set(g.nlargest(min(K,len(g)),"score").pos_id.astype(int))
                 for vid,g in E.groupby("veh_id")}
        r=solve_residual(
            uncommitted,positions,cfg,committed,time_limit=0.25,allowed=allowed
        )
        total+=r["runtime_s"]; stages+=1; final=r
        if r["success"] and r["deferred"]==0:
            break

    fallback=False
    if (not final["success"]) or final["deferred"]>0:
        fallback=True
        incumbent=final["objective"] if final["success"] else None
        fb=solve_residual(
            uncommitted,positions,cfg,committed,time_limit=0.60,
            allowed=None,cutoff=incumbent
        )
        total+=fb["runtime_s"]
        if fb["success"] and ((not final["success"]) or fb["objective"]<=final["objective"]+1e-9):
            final=fb

    return final,total,stages,fallback,"Adaptive-NGFLO",ph

def full_controller(uncommitted,positions,cfg,committed,congestion):
    r=solve_residual(uncommitted,positions,cfg,committed,time_limit=0.40)
    return r,r["runtime_s"],1,False,"Full-MIP",np.nan

def select_commitments(assignments, uncommitted, epoch, horizon, commit_fraction=0.45):
    """
    Commit a prefix of every occupied lane.

    This is required for physical consistency: once vehicles at the ramp-side
    front of a lane are committed, later arrivals may only be appended behind
    that fixed prefix. Committing an arbitrary interior vehicle would create
    an impossible hole in the lane sequence.
    """
    if assignments is None or len(assignments)==0:
        return pd.DataFrame(columns=assignments.columns if assignments is not None else [])

    frac=min(1.0,commit_fraction+0.18*epoch/max(1,horizon-1))
    blocks=[]
    for pid,g in assignments.groupby("pos_id"):
        g=g.sort_values("slot")
        k=max(1,int(round(frac*len(g)))) if len(g) else 0
        blocks.append(g.head(k))
    if not blocks:
        return assignments.iloc[0:0].copy()
    return pd.concat(blocks,ignore_index=True)



def audit_committed_plan(committed, positions, cfg, tol=1e-8):
    if len(committed)==0:
        return {
            "max_capacity_violation":0.0,
            "discharge_inversions":0,
            "duplicate_slots":0,
            "transverse_violation":0.0
        }
    pidx=positions.set_index("pos_id")
    maxv=0.0
    inversions=0
    duplicate_slots=int(committed.duplicated(subset=["pos_id","slot"]).sum())

    for pid,g in committed.groupby("pos_id"):
        pid=int(pid)
        maxv=max(
            maxv,
            max(0.0,float(g.length.sum())-float(pidx.loc[pid,"length_cap"])),
            max(0.0,float(g.mass.sum())-float(pidx.loc[pid,"mass_cap"]))
        )
        seq=g.sort_values("slot").destination.astype(int).tolist()
        inversions+=sum(seq[k]>seq[k+1] for k in range(len(seq)-1))

    for deck,g in committed.groupby("deck"):
        maxv=max(maxv,max(0.0,float(g.mass.sum())-cfg.deck_mass_capacity))

    total_mass=float(committed.mass.sum())
    moment=sum(
        float(r.mass)*float(pidx.loc[int(r.pos_id),"y_coord"])
        for _,r in committed.iterrows()
    )
    trans_v=max(0.0,abs(moment)-cfg.transverse_balance_ratio*total_mass)
    maxv=max(maxv,trans_v)
    return {
        "max_capacity_violation":float(maxv),
        "discharge_inversions":int(inversions),
        "duplicate_slots":int(duplicate_slots),
        "transverse_violation":float(trans_v)
    }


def run_episode(controller,arrival_regime,congestion,seed,horizon=4,cancel_prob=0.03):
    rng=np.random.default_rng(seed)
    cfg=base.FERRY_PRESETS["small"]
    positions=base.make_positions(cfg)

    waiting=pd.DataFrame(columns=[
        "veh_id","class","length","width","height","mass",
        "destination","priority","arrival_epoch"
    ])
    committed=pd.DataFrame(columns=[
        "veh_id","pos_id","slot","deck","lane","length","width","height","mass",
        "destination","priority","class","commit_epoch"
    ])

    next_id=0
    logs=[]
    total_arrivals=0
    total_cancellations=0
    cumulative_wait=0.0
    total_runtime=0.0
    new_commitments_total=0

    for t in range(horizon):
        # Only waiting/uncommitted vehicles may cancel.
        if len(waiting):
            cancel_mask=rng.random(len(waiting))<cancel_prob
            total_cancellations+=int(cancel_mask.sum())
            waiting=waiting.loc[~cancel_mask].reset_index(drop=True)

        na=arrival_count(rng,arrival_regime,t,horizon)
        arr=generate_arrivals(na,seed*100+t,congestion,next_id)
        next_id+=na; total_arrivals+=na
        if len(arr):
            arr["arrival_epoch"]=t
            waiting=pd.concat([waiting,arr],ignore_index=True)

        if len(waiting):
            if controller=="ngflo":
                r,rt,stages,fb,route,ph=adaptive_controller(
                    waiting,positions,cfg,committed,congestion
                )
            else:
                r,rt,stages,fb,route,ph=full_controller(
                    waiting,positions,cfg,committed,congestion
                )
        else:
            r={"success":True,"objective":0.0,"assignments":pd.DataFrame(),
               "loaded_ids":set(),"deferred":0,"mip_gap":0.0}
            rt=0.0; stages=0; fb=False; route="none"; ph=0.0

        total_runtime+=rt

        new_commits=select_commitments(
            r["assignments"],waiting,t,horizon,commit_fraction=0.45
        )
        if len(new_commits):
            new_commits=new_commits.copy()
            # Convert local residual slot indices to absolute append positions.
            if len(committed):
                prefix_counts=committed.groupby("pos_id").size().to_dict()
            else:
                prefix_counts={}
            new_commits["slot"]=[
                int(prefix_counts.get(int(pid),0))+int(k)
                for pid,k in zip(new_commits.pos_id,new_commits.slot)
            ]
            new_commits["commit_epoch"]=t
            committed=pd.concat([committed,new_commits],ignore_index=True)
            ids=set(new_commits.veh_id.astype(int))
            waiting=waiting[~waiting.veh_id.astype(int).isin(ids)].reset_index(drop=True)
            new_commitments_total+=len(ids)

        # Physically meaningful waiting: only uncommitted vehicles remain.
        if len(waiting):
            cumulative_wait += float(
                sum(t-int(a)+1 for a in waiting.arrival_epoch)
            )
            mean_wait=float(np.mean([t-int(a)+1 for a in waiting.arrival_epoch]))
        else:
            mean_wait=0.0

        lane_len,lane_mass,deck_mass,trans,occupied=committed_usage(committed,positions)
        logs.append({
            "epoch":t,"arrivals":na,"waiting_after_commit":len(waiting),
            "committed_total":len(committed),"new_committed":len(new_commits),
            "runtime_s":rt,"objective_uncommitted":r["objective"],
            "deferred_uncommitted":r["deferred"],"fallback":fb,"route":route,
            "hard_probability":ph,"mip_gap":r["mip_gap"],"mean_wait_age":mean_wait,
            "max_lane_length_utilization":max(
                lane_len[p]/float(positions.set_index("pos_id").loc[p,"length_cap"])
                for p in lane_len
            ) if lane_len else 0.0,
            "max_deck_mass_utilization":max(
                deck_mass[d]/cfg.deck_mass_capacity for d in deck_mass
            ) if deck_mass else 0.0,
        })

    # At departure, final attempt to load all remaining waiting vehicles without new commitment filtering.
    if len(waiting):
        if controller=="ngflo":
            r,rt,stages,fb,route,ph=adaptive_controller(
                waiting,positions,cfg,committed,congestion
            )
        else:
            r,rt,stages,fb,route,ph=full_controller(
                waiting,positions,cfg,committed,congestion
            )
        total_runtime+=rt
        final_assign=r["assignments"]
        if len(final_assign):
            final_assign=final_assign.copy()
            prefix_counts=committed.groupby("pos_id").size().to_dict() if len(committed) else {}
            final_assign["slot"]=[
                int(prefix_counts.get(int(pid),0))+int(k)
                for pid,k in zip(final_assign.pos_id,final_assign.slot)
            ]
            final_assign["commit_epoch"]=horizon
            committed=pd.concat([committed,final_assign],ignore_index=True)
            loaded_ids=set(final_assign.veh_id.astype(int))
            waiting=waiting[~waiting.veh_id.astype(int).isin(loaded_ids)].reset_index(drop=True)

    final_loaded=len(committed)
    final_deferred=len(waiting)

    # Dynamic cost: final deferral + waiting + runtime + commitment churn = none by design.
    dynamic_score = (
        120.0*final_deferred
        + 2.0*cumulative_wait
        + 0.05*total_runtime
    )

    audit=audit_committed_plan(committed,positions,cfg)
    summary={
        "controller":controller,"arrival_regime":arrival_regime,
        "congestion":congestion,"seed":seed,"horizon":horizon,
        "total_arrivals":total_arrivals,"total_cancellations":total_cancellations,
        "final_loaded":final_loaded,"final_deferred":final_deferred,
        "committed_total":len(committed),"cumulative_wait_units":cumulative_wait,
        "total_runtime_s":total_runtime,"fallback_count":sum(bool(x["fallback"]) for x in logs),
        "dynamic_score":dynamic_score,
        **audit
    }
    return summary,logs,committed,waiting


summaries=[]; logs=[]
INTENSITIES={"medium":1.00}
SEEDS=[101,202,303]
ARRIVAL_REGIME="poisson"

for intensity_name,scale in INTENSITIES.items():
    ARRIVAL_SCALE=scale
    for congestion in ["balanced","heavy","car_dense"]:
        for seed in SEEDS:
            for controller in ["full","ngflo"]:
                s,l,c,w=run_episode(
                    controller,ARRIVAL_REGIME,congestion,seed,horizon=4
                )
                s["intensity"]=intensity_name
                s["arrival_scale"]=scale
                summaries.append(s)
                for row in l:
                    row.update({
                        "controller":controller,
                        "arrival_regime":ARRIVAL_REGIME,
                        "congestion":congestion,
                        "seed":seed,
                        "intensity":intensity_name,
                        "arrival_scale":scale
                    })
                    logs.append(row)
                print(
                    intensity_name,congestion,seed,controller,
                    "loaded",s["final_loaded"],
                    "deferred",s["final_deferred"],
                    "runtime",round(s["total_runtime_s"],3),
                    "wait",s["cumulative_wait_units"]
                )

sdf=pd.DataFrame(summaries)
ldf=pd.DataFrame(logs)
sdf.to_csv(OUT/"publication_episode_summary.csv",index=False)
ldf.to_csv(OUT/"publication_epoch_log.csv",index=False)

# Pair Full-MIP and NGFLO on exactly the same stochastic path.
comp=sdf.pivot_table(
    index=["intensity","arrival_scale","congestion","seed"],
    columns="controller",
    values=[
        "dynamic_score","total_runtime_s","final_loaded","final_deferred",
        "cumulative_wait_units","fallback_count",
        "max_capacity_violation","discharge_inversions","duplicate_slots",
        "transverse_violation"
    ]
)
comp.columns=["_".join(c) for c in comp.columns]
comp=comp.reset_index()
comp["runtime_ratio_ngflo_full"]=comp["total_runtime_s_ngflo"]/comp["total_runtime_s_full"]
comp["runtime_change_pct"]=100*(comp["runtime_ratio_ngflo_full"]-1)
comp["score_change_pct"]=100*(comp["dynamic_score_ngflo"]-comp["dynamic_score_full"])/np.maximum(
    1e-9,np.abs(comp["dynamic_score_full"])
)
comp["loaded_difference"]=comp["final_loaded_ngflo"]-comp["final_loaded_full"]
comp["deferred_difference"]=comp["final_deferred_ngflo"]-comp["final_deferred_full"]
comp["wait_difference"]=comp["cumulative_wait_units_ngflo"]-comp["cumulative_wait_units_full"]
comp.to_csv(OUT/"publication_paired_results.csv",index=False)

# Aggregate across seeds.
agg=comp.groupby(["intensity","arrival_scale","congestion"]).agg(
    pairs=("seed","count"),
    loaded_diff_mean=("loaded_difference","mean"),
    loaded_diff_min=("loaded_difference","min"),
    deferred_diff_mean=("deferred_difference","mean"),
    wait_diff_mean=("wait_difference","mean"),
    runtime_change_pct_mean=("runtime_change_pct","mean"),
    runtime_change_pct_median=("runtime_change_pct","median"),
    score_change_pct_mean=("score_change_pct","mean"),
    score_change_pct_median=("score_change_pct","median"),
    full_runtime_mean=("total_runtime_s_full","mean"),
    ngflo_runtime_mean=("total_runtime_s_ngflo","mean"),
    full_loaded_mean=("final_loaded_full","mean"),
    ngflo_loaded_mean=("final_loaded_ngflo","mean"),
    full_deferred_mean=("final_deferred_full","mean"),
    ngflo_deferred_mean=("final_deferred_ngflo","mean"),
    full_wait_mean=("cumulative_wait_units_full","mean"),
    ngflo_wait_mean=("cumulative_wait_units_ngflo","mean"),
    ngflo_fallback_mean=("fallback_count_ngflo","mean"),
).reset_index()
agg.to_csv(OUT/"publication_aggregate_results.csv",index=False)

# Overall feasibility audit.
feas={
    "episodes":int(len(sdf)),
    "pairs":int(len(comp)),
    "max_capacity_violation":float(sdf.max_capacity_violation.max()),
    "max_discharge_inversions":int(sdf.discharge_inversions.max()),
    "max_duplicate_slots":int(sdf.duplicate_slots.max()),
    "max_transverse_violation":float(sdf.transverse_violation.max()),
}
feas["all_feasible"]=bool(
    feas["max_capacity_violation"]<=1e-8
    and feas["max_discharge_inversions"]==0
    and feas["max_duplicate_slots"]==0
    and feas["max_transverse_violation"]<=1e-8
)
(OUT/"publication_feasibility_audit.json").write_text(
    json.dumps(feas,indent=2),encoding="utf-8"
)

print("\\nAGGREGATE")
print(agg.to_string(index=False))
print("\\nFEASIBILITY")
print(json.dumps(feas,indent=2))
