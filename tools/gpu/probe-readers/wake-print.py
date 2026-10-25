import json, sys
# wake-print.py A B: per step, particles, h tiles, regions by heat, retired/seeded counters and region tiers for two probe logs.
def load(f):
    steps, heat = {}, {}
    for l in open(f):
        if l.startswith('{"step"'):
            d = json.loads(l)
            steps[d['step']] = d
        elif l.startswith('{"heatState"'):
            d = json.loads(l)
            heat[d['heatState']] = {t: len(rs) for t, rs in d['regions'].items()}
    return steps, heat
(sa, ha), (sb, hb) = load(sys.argv[1]), load(sys.argv[2])
def row(d):
    r = d.get('retired', {})
    return 'p %6s hT %5s reg %s seeded %s' % (d.get('particles'), d.get('tiles', {}).get('h'), d.get('regions'), r.get('splitSeeded'))
for s in sorted(set(sa) & set(sb)):
    if s < 24:
        continue
    print(s, 'A', row(sa[s]), ha.get(s))
    print(s, 'B', row(sb[s]), hb.get(s))
