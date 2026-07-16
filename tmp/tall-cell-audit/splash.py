import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"): continue
    r = json.loads(line)
    if r.get("phase") != "running" or r["method"] != "tall-cell": continue
    post = r["postProjectionVelocity"]
    if post["liquidMaximum"] < 8: continue
    ex = r.get("extrema") or {}
    ms = ex.get("maxSpeed") or {}
    gaps = r.get("tallVolumeGaps") or {}
    act = r.get("tallCellActivity") or {}
    print("s%3d t=%.2f liqMax=%6.1f cfl=%5.2f loc=%s postLoc=%s ceil=%s mixed=%s minBase=%s hist_lo=%s" % (
        r["steps"], r["simulatedTime_s"], post["liquidMaximum"], post["maximumComponentCfl"],
        (ms.get("x"), ms.get("y"), ms.get("z"), ms.get("region")), post["location"],
        gaps.get("wetBandCeilingColumns"), gaps.get("mixedEndpointColumns"),
        act.get("minimumTallHeight"), {k: v for k, v in sorted((act.get("histogram") or {}).items(), key=lambda kv: int(kv[0]))[:4]}))
