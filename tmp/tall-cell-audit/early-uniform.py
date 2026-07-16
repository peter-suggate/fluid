import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"): continue
    r = json.loads(line)
    if r.get("phase") != "running" or r["method"] != "uniform" or r["steps"] > int(sys.argv[2]):
        continue
    pre, post = r["preProjectionVelocity"], r["postProjectionVelocity"]
    ratio = post["kineticEnergyProxy"] / max(pre["kineticEnergyProxy"], 1e-30)
    print("s%3d preMax=%.3f postMax=%.3f keRatio=%.3f loc=%s comp=%d" % (
        r["steps"], pre["maximum"], post["maximum"], ratio, post["location"], post["component"]))
