# Dynamic pressure-topology diagnosis

## Result

The dynamic-scene spike is GPU serialization, not CPU allocation or host
scheduling. In the matched long-dam B16/P16 trace, a pressure-topology stage
following a topology commit costs 106.9548 ms median versus 2.2282 ms on a
quiescent input. The matched command-encoding intervals are 0.1808 ms and
0.1504 ms respectively.

The complete receipt is
`artifacts/sparse-cm12-pressure-topology-long-dam-work-trace.json`.

## Evidence ranking

1. `classifyPressureTopologyCells` is a one-invocation shader. It walks every
   physical brick, then serially classifies every old and new cell of each
   changed B16 brick. One brick can contribute 4096 old and 4096 new cells.
2. `classifyPressureTopologyRows` is also a one-invocation shader. For every
   old/new changed cell it serially walks the complete incidence range and
   recomputes row membership/theta. Its generation stamp deduplicates writes,
   but does not parallelize the incidence traversal.
3. `finalizePressureTopologyRepair` is a third singleton loop over every
   physical brick, even on a quiescent frame.
4. After membership, the stage still runs global effective-edge, brick
   aggregate, and hierarchy edge/diagonal bakes. These explain most of the
   approximately 2.2 ms quiescent floor but not the 107 ms dynamic spike.
5. Host work is fixed: 22 pressure-topology dispatches, six compute passes and
   five storage-to-indirect copies are encoded every frame. No dynamic count is
   read by the CPU. The resident advance creates no per-frame pressure buffer,
   pipeline, bind group, typed array, map, or topology-dependent host list.
   The only per-frame `Set` in `encode` is pressure-journal scheduling and is
   absent unless the QA journal is armed.
6. Dynamic topology preparation/commit is GPU encoded. Its full template
   cell/row dispatches and singleton commit loops belong to resolution planning
   and candidate transfer, not the measured pressure-topology stage.

The trace observed three slow pressure inputs. Each followed a frame that
committed only two bricks:

| Input | Accepted cells/rows | Temporal cells/rows | PCM dirty cell/row leaves |
| --- | ---: | ---: | ---: |
| changed 1 | 51,328 / 149,232 | 47,420 / 139,542 | 238 / 729 |
| changed 2 | 51,328 / 149,232 | 46,795 / 137,660 | 208 / 643 |
| changed 3 | 44,160 / 127,920 | 39,880 / 117,048 | 210 / 660 |

Those parallel-domain counts became smaller while the stage became about 48
times slower. This rules out ordinary pressure-cell dispatch volume and points
directly to the serial old/new topology walks.

## Timestamp attribution correction

The hardware stage boundary itself is exact: the stage begins after body force,
includes all five copy seams and six pressure-topology passes, and closes before
pressure RHS. The hardware phase partition closes with zero error.

The old changed/quiescent diagnostic label was off by one frame. Topology is
committed after pressure/projection, so a frame's terminal committed-brick
count describes topology that pressure will consume on the next frame. The
probe now buckets pressure topology using the prior frame's commit and emits
`pressureTopologyInputChangedPerFrame` plus exact work receipts. Other stages
retain same-frame terminal bucketing.

## Required production repair

Topology publication must append one generation-stamped changed-brick record
containing stable brick ID, old state/range, new state/range, topology
generation and complete provenance. Queue order may be nondeterministic because
it only marks canonical stable-ID leaves.

Pressure repair then uses:

1. one 64-lane workgroup per changed B16 brick;
2. lane-strided old/new cell ranges to set PCM cell candidates;
3. incidence writes that mark generation-stamped stable row leaves, without
   classifying duplicate rows in the brick workgroup;
4. canonical row-leaf rank/select, classifying each stable row once;
5. exact PCM transition events feeding PCF fine edge/diagonal repair;
6. immutable owner maps propagating changed fine values only to brick
   aggregate and hierarchy ancestors;
7. PSA wet-brick and active-node repair from the same accepted PCM/PCF tuple.

The finalizer may be one invocation only if it reads compact headers, root
counts and generation receipts. It must update stored brick state only for the
changed-brick records. It may not loop over bricks, cells, rows, edges or
hierarchy nodes.

Every work family uses a GPU-authored copy-isolated indirect triplet. Capacity,
generation or provenance failure zeros the affected indirects and leaves the
candidate generation unaccepted; it does not launch a global classifier or
coefficient bake.

The serial integration gate is one paired five-step dam run comparing physical
fields plus PCM membership/theta, PCF coefficients/diagonals/RHS and PSA
wet-brick/node authority byte-for-byte. After that passes, ocean timing must
show pressure topology below 1 ms on quiescent B16/P16 and dynamic work
proportional to the changed-brick/row-leaf/ancestor receipts. The combined
milestone, rather than each internal authority, receives the next 60-step gate.

