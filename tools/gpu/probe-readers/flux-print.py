import json, sys
# flux-print.py LOG: per heatState sample, the surface regions' freeze terms by tier (max and median), and the floor r dt.
for l in open(sys.argv[1]):
    if not l.startswith('{"heatState"'):
        continue
    d = json.loads(l)
    for tier, rs in d['regions'].items():
        out = {}
        for k in ('surf', 'mom', 'pres', 'froz', 'disp'):
            v = sorted(r[k] for r in rs)
            out[k] = '%.2e/%.2e' % (v[len(v) // 2], v[-1])
        trig = sorted(set(r['trig'] for r in rs))
        print(d['heatState'], tier, len(rs), out, 'trig', trig[:8])
