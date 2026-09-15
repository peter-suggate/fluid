# Half-pool phi vertex failure at frame 13

The `coarse-first-pool-impact-half` Dawn reproduction halted in
`lsvInsertVertexHash`, generation 14, with fault mask 8 (duplicate vertex /
malformed constraint). This was not hash-capacity exhaustion. Duplicated
coordinates included `(28,24,8)` and `(32,24,4)`.

Fixed host guard pages have template entries for unavailable rungs that alias
their accepted cells. `cm12PhiBrickPlan` checked only that the fine entry had a
nonzero cell count. When the expanding interface band reached a fixed B2 page,
the plan labelled its B2 geometry as B8. Coordinate-to-cell lookup then divided
positions by the wrong cell width, disagreeing with the geometry used to emit
corners. Adjacent cells could both claim ownership of one vertex.

The planner now checks the stored cell resolution before selecting B8. An
aliased entry keeps the accepted coarse rung; actual prepared B8 entries still
select B8. The fix adds a metadata check during planning, no new topology
construction, search expansion, or repair. Duplicate-vertex failure remains
fatal.

The focused Dawn reader fixture exercises the production planner with both an
aliased B2 entry and a real B8 entry. The scene diagnostic now indexes the
64-by-48-by-64 domain using its actual dimensions rather than assuming a cube.

## Receiver failure after the vertex fix

The next failure was fault 21 in `addWholeFrameUncoveredDonorFallbacks`.
The apparent missing receiver was an active B2 page next to a B8 page: the
accepted image had an unsupported 4:1 seam. Candidate planning had counted
requests that fixed pages could not publish. Coarse-first staging could also
request B1 for a fixed B2 page, invalidating neighboring plans while other
transitions proceeded.

Fixed pages now retain their accepted rung both in ordinary planning and in
frontier activation. Their actual prepared maximum is a hard grading cap,
propagated through the existing closure dispatches even without authored
refinement regions. Neighbor refinement therefore respects the geometry that
will actually be published. Mutable pages still admit at the coarsest graded
rung. No runtime topology generation or repair was added.

An extension-order trial was removed: the final patch keeps the original
velocity-extension order and dispatch count.

## Validation

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx tools/probe-figure7-deformation-dawn.ts \
  --scene=coarse-first-pool-impact-half --steps=90 --verify-health \
  --failure-debug --output=artifacts/level-set-volume/half-pool-final-3s.json
```

The scene reaches 3 seconds (90 frames). All 91 checkpoints, including the
initial state, pass activity, topology-commit, transport, coupling-capacity,
and finite/nonnegative-density checks. GPU validation errors are empty.
Recorded residue deletion is 22.124533 finest-cell volumes; remaining volume
plus recorded deletion differs from the initial 69724 by 0.383335 finest-cell
volumes (about 0.00055%). This is a correctness smoke check, not a new visual
quality or performance baseline.

The focused `sparse-cm12-coarse-readers-dawn.test.ts` GPU fixture checks aliased
versus genuine fine geometry, fixed versus mutable admission, and prepared
grading caps, alongside existing sparse-air and overlay checks.
