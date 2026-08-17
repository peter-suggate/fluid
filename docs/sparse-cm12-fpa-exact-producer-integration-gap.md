# FPE1 → FPA1 live integration-gap ledger

Status: standalone adapter and immutable CPU/Naga contract exist; the resident
does not import or allocate FPE1. Production remains unchanged.

The sibling FTC1 observational census now applies
`fpaPreparationRowLive(row)` before full/tile membership and witness tracking;
raw incidence-visit counters remain raw. FPE1 uses the same predicate.

FPE1 is producer ingress only. It calls `fpaMarkPreparationRow` and
`fpaCoverPreparationReceipt`; FPA1 remains the sole row work/result authority.
There is no accepted-cell/row scan and no fallback dispatch.

| Producer family | Exact evidence required | Deterministic expansion | Live gap |
|---|---|---|---|
| Liquid phase cell | Previous/current liquid-phase bits after final density/gamma publication | Immutable accepted cell incidence | Temporal classification currently broadens cells into physical dirty bricks; it does not replay exact per-cell change receipts into FPA collection. |
| Accepted VEX cell | Previous/current accepted xyz plus validity bits at VEX commit | Compiled reverse VEX-sample → preparation-row CSR | VEX retains accepted values, but no reverse preparation dependency CSR or replay worklist is wired. Incidence alone is insufficient for traced interpolation. |
| Final source-face row | Previous/current final projected source-face u32 | Direct stable row | Projection does not publish an exact changed-row journal for the next frame’s preparation blend. |
| Topology cell/rung | Previous/current stable owner+rung identity; old and new cell events when identity changes | Immutable old/new incidence filtered by `fpaPreparationRowLive` | Topology roots exist for VEX and pressure repair, but no FPE1 replay receipts cover preparation rows. Retired rows must be filtered by FPA’s live predicate. |
| Moving-solid cell | Previous/current cell open/coverage/solid sample bits | Immutable accepted incidence | Current activity can observe collocated velocity changes, but exact solid cell input changes are not an FPA producer family. |
| Moving-solid row | Previous/current area/open-fraction/solid-velocity bits | Direct stable row | Rigid row-data publication has no exact changed-row replay into FPA collection. |
| Policy epoch row | Previous/current policy epoch plus explicit affected stable row | Direct stable row supplied by policy owner | `fpaPolicyBits()` records the current policy but does not compare prior policy or publish affected rows. FPE1 intentionally provides no global policy scan. |

## Required integration order

1. Late frame N producers append their exact stable cell/row ids to bounded,
   generation-stamped producer worklists and publish per-family counts.
2. Frame N+1 begins FPA1 preparation, then begins FPE1 with those device-owned
   expected counts.
3. Family replay dispatches call the corresponding `fpeRecord*` function.
   Cell families expand only through immutable incidence/reverse-dependency CSR;
   row families mark their supplied stable row directly.
4. Seal FPE1 before finalizing the FPA preparation frontier. Any generation,
   receipt, mark, overflow, or uncovered-write fault leaves FPE1 unaccepted and
   withholds the matching FPA receipt, so FPA also fails closed.
5. Accept FPE1 only after its sealed receipt is captured for QA. FPA alone plans,
   executes, verifies, mirrors, and accepts face results.

## Missing resident resources and hooks

- A bounded producer worklist per family, with accepted/candidate generation and
  indirect triplets. FPE1 deliberately does not manufacture these from scans.
- `fpeaExpectedProducerReceipts(family)` sourced from those sealed GPU lists.
- Immutable cell incidence hooks for liquid/topology/moving-cell replay.
- A compiled reverse VEX dependency CSR for the fixed preparation policy/CFL
  contract. This is the semantic blocker to treating one-ring incidence as exact.
- Exact changed-row journals from final face projection and rigid row-data
  publication.
- A policy-owner affected-row journal; changing a global epoch without such a
  journal must fault rather than silently reuse preparation authority.
- FPA expected-receipt composition must use the sum of FPE1 family counts, not
  the current dirty-physical-brick receipt count.
- Same-step QA receipt exposure for per-family expected/covered counts, first
  fault family/id, pending events, row writes, and cause mask.

## Static boundary

Run:

```bash
node --import tsx tools/check-sparse-cm12-fpa-exact-producer-adapter.ts
```

The checker covers all seven producer families, stable ascending row union,
exact-bit rejection, generation mismatch, invalid row, coverage gap, required
FPA live-row filtering, source-manifest constraints, and standalone Naga.
