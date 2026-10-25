import json, sys
# wake-diff.py A B STEP...: regions whose tier differs at each step, with A's and B's evidence.
def load(f):
    out = {}
    for l in open(f):
        if l.startswith('{"heatState"'):
            d = json.loads(l)
            out[d['heatState']] = {r['r']: (t, r) for t, rs in d['regions'].items() for r in rs}
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
for step in map(int, sys.argv[3:]):
    A, B = a.get(step, {}), b.get(step, {})
    rows = []
    for rid in sorted(set(A) | set(B)):
        ta = A.get(rid, ('-', {}))[0]; tb = B.get(rid, ('-', {}))[0]
        if ta != tb:
            ra = A.get(rid, ('-', {}))[1]; rb = B.get(rid, ('-', {}))[1]
            rows.append((rid, ta, tb, ra.get('trig'), rb.get('trig'), ra.get('surf'), ra.get('froz'), ra.get('c'), ra.get('fine')))
    print('step', step, 'differ', len(rows), 'A listed', len(A), 'B listed', len(B))
    for r in rows[:30]:
        print('  ', r)
