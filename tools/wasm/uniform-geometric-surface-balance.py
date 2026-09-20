"""Practical 2D surface-deficit study. Python stdlib; run from the repo root.
Use --soak to regenerate the 300-second runs as well. No production defaults change.
"""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import statistics
import subprocess

args=argparse.ArgumentParser()
args.add_argument("--soak",action="store_true")
args.add_argument("--out",default="docs/research/uniform-geometric-surface-balance-2026-09-20")
opts=args.parse_args()
root=Path(opts.out);root.mkdir(parents=True,exist_ok=True)
def load(path):
    with gzip.open(path,"rt") as f:return json.load(f)
def save(name,value):
    with gzip.open(root/(name+".json.gz"),"wt") as f:json.dump(value,f,separators=(",",":"))
base=load("docs/research/uniform-geometric-compensation-2026-09-20/baseline-input.json.gz")
subprocess.run(["cargo","build","--release","--manifest-path","rust/Cargo.toml","-p","fluid-core","--example","uniform_geometric_scene"],check=True)
def config(mode):
    return {"mode":"off"} if mode=="baseline" else {"mode":"balance-surface-deficit","balanceScope":"liquid-components" if mode=="components" else "global"}
def run(name,request,store=True):
    r=subprocess.run(["rust/target/release/examples/uniform_geometric_scene"],input=json.dumps(request),text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    result=json.loads(r.stdout)
    if store:save(name+"-input",request);save(name,result)
    print(name,round(result["elapsedMs"],1),"ms",flush=True)
    return result
if opts.soak:
    for mode in ["baseline","global"]:
        run("impact-"+mode+"-300",{**base,"frames":9000,"energyExperiment":config(mode)})

# Visual comparison: both the impact and the motion remaining after 300 seconds.
for mode in ["baseline","global"]:
    for part in ["early","late"]:
        q={**base,"frames":300,"energyExperiment":config(mode),"energySnapshotFrames":list(range(2,301,2)),"energySnapshotStages":["projected"]}
        if part=="late":
            state=load(root/("impact-"+mode+"-300.json.gz"))
            for key in ["volume","phi","velocity","lowX","lowY","released"]:q[key]=state[key]
        run("visual-"+part+"-"+mode,q)

# Warm the executable, then alternate order. No energy audit, source receipts,
# field snapshots, or extra pressure solves are included in the timed candidate.
timings={"baseline":[],"global":[]}
for mode in timings:
    run("warm-"+mode,{**base,"frames":60,"auditEnergy":False,"energyExperiment":{**config(mode),"auditReceipt":False}},False)
for repeat in range(5):
    for mode in (["baseline","global"] if repeat%2==0 else ["global","baseline"]):
        q={**base,"frames":600,"auditEnergy":False,"energyExperiment":{**config(mode),"auditReceipt":False}}
        r=run("timing-"+mode+"-"+str(repeat),q,False)
        timings[mode].append(r["elapsedMs"]/q["frames"])
save("timing",{"msPerFrame":timings,"median":{k:statistics.median(v) for k,v in timings.items()},"framesPerRun":600,"repeats":5,"auditEnergy":False,"auditReceipt":False})

flat={**base,"dimensions":[32,24],"cellSize":[.1,.1],"phi":[y*.1-1.2 for y in range(25) for x in range(33)],"volume":[float(y<12) for y in range(24) for x in range(32)],"capacity":[1.0]*(32*24)}
# Actual live-insertion path: enqueued LiquidDrop, consumed during ordinary advance.
for mode in ["baseline","global"]:
    q={**flat,"frames":300,"energyExperiment":config(mode),"liquidInjections":[{"frame":31,"drop":{"centre_m":[1.6,1.65],"radius_m":.35}}],"energySnapshotFrames":[30,31,60,150,300],"energySnapshotStages":["projected"]}
    r=run("injection-"+mode,q)
    injected=sum(x["injectedVolume"] for x in r["receipts"])
    assert injected>30
    assert abs(r["receipts"][-1]["volume"]-r["initialVolume"]-injected)<.05

# Two disjoint rectangular pools at zero gravity, paired surplus and deficit.
def box_phi(x,y,left,right):
    dx=max(left-x,x-right);dy=y-1.2
    return math.hypot(max(dx,0),max(dy,0))+min(max(dx,dy),0)
split={**flat,"gravity":[0,0],"frames":1,
    "options":{**base["options"],"densitySharpening":"off","totalSurfaceVolume":"off"},
    "phi":[min(box_phi(x*.1,y*.1,.2,1.4),box_phi(x*.1,y*.1,1.8,3.0)) for y in range(25) for x in range(33)],
    "volume":[(1.05 if 2<=x<14 else .95 if 18<=x<30 else 0.0) if y<12 else 0.0 for y in range(24) for x in range(32)],
    "energySnapshotFrames":[1],"energySnapshotStages":["sharpened","projected"]}
for mode in ["baseline","global","components"]:
    run("split-"+mode,{**split,"energyExperiment":config(mode)})

# Source-map capture at a late state: no time/history inference from restart.
state=load(root/"impact-global-300.json.gz")
q={**base,"frames":1,"energyExperiment":config("global"),"energySnapshotFrames":[1],"energySnapshotStages":["sharpened","projected"]}
for key in ["volume","phi","velocity","lowX","lowY","released"]:q[key]=state[key]
run("late-sources",q)

# Turn instrumentation off on identical inputs and prove fields are unchanged.
audited=run("audit-check",{**base,"frames":60,"energyExperiment":config("global")},False)
plain=run("plain-check",{**base,"frames":60,"auditEnergy":False,"energyExperiment":{**config("global"),"auditReceipt":False}},False)
for key in ["volume","phi","velocity","pressure","lowX","lowY","released","receipts"]:assert audited[key]==plain[key],key
save("checks",{"auditFieldsIdentical":True,"injectionMassChecks":True,"scope":"Native CPU 2D, static scene and live liquid insertion; no browser/GPU timing claim."})
