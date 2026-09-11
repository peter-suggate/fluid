"""Read-only CPU work-domain proof for the saved production Figure 7 census.
No simulation or GPU code is modified. Output counts, not predicted milliseconds.
"""
import json
from collections import deque
from pathlib import Path

root = Path(__file__).resolve().parents[1]
census = json.loads((root / "artifacts/cm12-figure-7-radius-01/cpu-census.json").read_text())["initial"]
assert all(b["resolution"] == 8 and b["span"] == 1 and b["active"] for b in census["bricks"])
points = set()
for b in census["bricks"]:
    ox, oy, oz = (8 * v for v in b["coordinate"])
    points.update((ox+x, oy+y, oz+z) for z in range(8) for y in range(8) for x in range(8))
seeds = {tuple(c[:3]) for c in census["wetCells"] if c[3] > .5}
steps = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
def neighbours(p):
    return [q for dx,dy,dz in steps if (q := (p[0]+dx,p[1]+dy,p[2]+dz)) in points]
adjacency = {p: neighbours(p) for p in points}
packets = {}
for p in points:
    packets.setdefault(tuple(v//4 for v in p), set()).add(p)
assert len(points) == census["cells"] == 18432
assert len(packets) == 288 and all(len(v)==64 for v in packets.values())
# Independent graph-distance authority; no assumption that Euclidean distance
# or wet-packet membership is enough to choose an extension work domain.
distance = {p:0 for p in seeds}
queue = deque(seeds)
while queue:
    p = queue.popleft()
    if distance[p] == 8: continue
    for q in adjacency[p]:
        if q not in distance:
            distance[q] = distance[p]+1
            queue.append(q)
reachable_packets = {tuple(v//4 for v in p) for p in distance}
selected = set().union(*(packets[p] for p in reachable_packets))
# A cheaper conservative selector: inflate each occupied seed packet's box
# by eight graph steps. This overselects because the precise seed lanes are
# discarded. Only a candidate cost model; no GPU selector has been implemented.
seed_packets = {tuple(v//4 for v in p) for p in seeds}
def box_gap(a,b):
    return sum(max(0,4*abs(x-y)-3) for x,y in zip(a,b))
conservative_packets = {p for p in packets if any(box_gap(p,s)<=8 for s in seed_packets)}
conservative = set().union(*(packets[p] for p in conservative_packets))
assert selected <= conservative

# Compare exhaustive dense validity recurrence with restricted packet traversal,
# including every valid cell at every intermediate depth, not just final counts.
def recurrence(domain):
    valid = set(seeds)
    trace=[]
    for depth in range(1,9):
        attempts=domain-valid
        neighbor_reads=sum(len(adjacency[p]) for p in attempts)
        additions={p for p in attempts if any(q in valid for q in adjacency[p])}
        valid |= additions
        expected={p for p,d in distance.items() if d<=depth}
        assert valid == expected, (depth,len(valid),len(expected))
        trace.append(dict(depth=depth,validCells=len(valid),newCells=len(additions),
                          attemptedCells=len(attempts),neighborDepthReads=neighbor_reads))
    return trace
arms={"accepted":recurrence(points),"exactEightHopPackets":recurrence(selected),
      "conservativeSeedPacketBoxes":recurrence(conservative)}
report={"scope":"Figure 7 radius 0.1 m production reset; B8, no mixed seams, 32 liquid seeds, eight sweeps",
        "proof":"Dense and both restricted validity recurrences agree at every depth; numerical order within retained cells is unchanged.",
        "acceptedCells":len(points),"seedCells":len(seeds),"acceptedPackets":len(packets),
        "reachableCells":len(distance),"exactEightHopPackets":len(reachable_packets),
        "conservativePackets":len(conservative_packets),"seedPackets":len(seed_packets),
        "arms":{}}
for name, domain in [("accepted",points),("exactEightHopPackets",selected),("conservativeSeedPacketBoxes",conservative)]:
    report["arms"][name]={"initializationCellVisitsRetained":len(points),
        "sweepCellInvocations":8*len(domain),"sweepWorkgroups":8*len(domain)//64,
        "neighborDepthReads":sum(x["neighborDepthReads"] for x in arms[name]),"perDepth":arms[name]}
# The current direct rebuild frame redundantly clears each invalid packet's
# alternating output mask in all eight sweeps. A separate initial scrub must
# establish both mask banks before sweeps may consume only accepted packets.
capacity_packets=(36+census["pool"]["pageCapacity"])*8
report["rebuildProof"]={"capacityPackets":capacity_packets,"acceptedPackets":len(packets),
    "currentNinePassWorkgroups":9*capacity_packets,
    "oneDirectInitializationEightAcceptedSweeps":capacity_packets+8*len(packets),
    "workgroupsRemoved":8*(capacity_packets-len(packets)),
    "laneInvocationsRemoved":64*8*(capacity_packets-len(packets)),
    "invalidPacketMaskStoresBefore":16*(capacity_packets-len(packets)),
    "additionalInitialMaskStoresRequired":2*(capacity_packets-len(packets)),
    "invalidMaskStoresNetRemoved":14*(capacity_packets-len(packets))}
# Stale-state proof for the simple rebuild change. Start the second bank with
# adversarial old bits. Keep the production direct initializer and add only
# the missing invalid-packet B clear. After the first sweep both banks agree;
# all later invalid-packet overwrites in the baseline are redundant zeros.
from array import array
valid_count=2*len(packets)
capacity_words=2*capacity_packets
old_a=array('I',[0])*capacity_words
new_a=array('I',[0])*capacity_words
old_b=array('I',[(i*2654435761)&0xffffffff for i in range(capacity_words)])
new_b=array('I',old_b)
for i in range(valid_count,capacity_words): new_b[i]=0
for depth in range(1,9):
    old=old_b if depth&1 else old_a
    new=new_b if depth&1 else new_a
    for i in range(capacity_words):
        # Any accepted result is permitted: proof does not depend on zero
        # velocity, full packets, scalar values, or a particular mask pattern.
        value=((i+depth)*2246822519)&0xffffffff if i<valid_count else 0
        old[i]=value
        if i<valid_count: new[i]=value
    assert old_a==new_a and old_b==new_b
report["rebuildProof"]["staleMaskBankEquivalenceAfterEverySweep"] = True
# Existing hardware receipt: count topology-generation transitions, not the
# narrower pressure-row-change attribution, which misses other topology changes.
receipt=json.loads((root/"artifacts/cm12-figure-7-radius-01/pressure-row-load-final.json").read_text())
prior=None; rebuild_frames=[]
for sample in receipt["diagnostic"]["authoritySamples"]:
    generation=sample["finalScalarMasks"]["topologyGeneration"]
    if generation!=prior:rebuild_frames.append(sample["advance"])
    prior=generation
report["existingDawnEvidence"]={"source":"pressure-row-load-final.json",
    "topologyGenerationTransitionFrames":rebuild_frames,
    "note":"Transitions guarantee schedule rebuilds; slot-only changes could add more. No per-frame accepted-packet readback was captured.",
    "terminalVexHeader":receipt["velocityExtension"]}
out=root/"artifacts/cm12-figure-7-radius-01/small-fluid-work-proof.json"
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="arms"},indent=2))
for k,v in report["arms"].items(): print(k,{a:b for a,b in v.items() if a!="perDepth"})
