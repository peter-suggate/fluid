import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"): continue
    r = json.loads(line)
    if r.get("phase") != "running" or r["method"] != "tall-cell" or r["steps"] > int(sys.argv[2]):
        continue
    pre, post = r["preProjectionVelocity"], r["postProjectionVelocity"]
    ex = r.get("extrema") or {}
    ms = ex.get("maxSpeed") or {}
    ratio = post["kineticEnergyProxy"] / max(pre["kineticEnergyProxy"], 1e-30)
    loc = (ms.get("x"), ms.get("y"), ms.get("z"), ms.get("region", ""))
    print("s%3d preMax=%.3f postMax=%.3f keRatio=%.2f divB=%.1f divA=%.1f residRel=%.3f speedLoc=%s" % (
        r["steps"], pre["maximum"], post["maximum"], ratio,
        r["maxDivergenceBefore_s"], r["maxDivergenceAfter_s"],
        r["pressureRelativeResidual"], loc))
