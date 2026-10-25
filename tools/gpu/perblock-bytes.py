import json, re, sys
# tools/gpu/perblock-bytes.py <audit a> <blocks a> <audit b> <blocks b>: per-resource MiB in each and the marginal MiB a block.
def load(p):
    for line in open(p):
        if line.startswith('{"audit":"end"'):
            d = json.loads(line)
            out = {}
            for r in d["resources"]:
                m = re.match(r"([0-9.]+) (.*)", r)
                out[m.group(2)] = float(m.group(1))
            return d["liveMiB"], out
    raise SystemExit("no audit in " + p)
la, a = load(sys.argv[1]); na = int(sys.argv[2])
lb, b = load(sys.argv[3]); nb = int(sys.argv[4])
rows = []
for k in set(a) | set(b):
    x, y = a.get(k, 0.0), b.get(k, 0.0)
    rows.append(((y - x) / (nb - na), x, y, k))
rows.sort(reverse=True)
print(f"live {la:.1f} -> {lb:.1f} MiB; marginal {1024*(lb-la)/(nb-na):.1f} KiB a block")
for per, x, y, k in rows:
    if abs(per) * 1024 >= 1 or x >= 5:
        print(f"{per*1024:8.1f} KiB/blk  {x:8.1f} {y:8.1f}  {k}")
