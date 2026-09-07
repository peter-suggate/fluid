# Half-pool lifecycle failures: directory and membership repair

The supplied `coarse-first-pool-impact-half` receipt reproduces exactly with
its authored configuration: frame 42, generation 44, owner 29285,
`EMPTY_DEFICIT_STENCIL`, visible weight zero and deficit one.

The donor is centred at `(31.5, 47.5, 30.5)` in finest-cell coordinates. At the
end of step 42 its density is `0.000037219459045445547` and velocity is
approximately `(0.215914, 51.514534, 0.444629)` finest cells per second. This is
real dilute mass travelling upward through the open top, not an empty-air
trace. Source page `(3,5,3)` requests receiver `(3,6,3)` in both support masks.
The receiver is absent, while adjacent receivers `(4,6,3)`, `(3,6,4)` and
`(4,6,4)` are active.

## Cause

The frontier's per-source resolved-direction cache (activity word 45) retained
an obsolete positive directory lookup. All four upper pages had been allocated
at step one and retired at step two. When the splash later requested them
again, the source's cached answer for `(3,6,3)` still said the request was
resolved, so allocation skipped it despite the explicit transport demand.

Invalidation lived in the simulation candidate-commit path, but the actual
world-directory deletion belongs to presentation retirement, after the all-air
publication. These are different lifetimes. Clearing a cache before deletion
allows another lookup to cache the still-present key; tying it only to that
commit path also fails to cover every way the eventual release is reached.
Additional diagnostic publication/readback changed the observed trajectory,
which is why the permanent regression advances the ordinary solver directly.

## Repair

Invalidate surviving neighbours' reciprocal direction bits immediately after
`cm12WorldReleaseLeaf` succeeds in `retireSparseCM12PresentationPages`. Remove
the earlier candidate-commit invalidation. The release and cache invalidation
now have one authority and execute before the next frontier allocation pass.
The invalidation is bounded to 26 neighbours of a released page; no additional
GPU pass, density cutoff, donor return, or allocation-budget increase is used.

The failure receipt now includes donor density in the previously reserved
third operand. The existing failure guard still rejects every nonzero donor,
including density below the numerical dry threshold.

The receipt-driven probe now applies the supplied method overrides and uses
the scene timestep for its target times, with an optional capture-start step.

## Regression

`tests/sparse-cm12-half-pool-deficit-dawn.test.ts` retains the supplied scene and
method configuration as a fixture. It runs all 120 paper steps, checks health,
finite nonnegative density, finite velocity, allocation status and mass drift
below 0.1% on every step, and
requires the missing receiver to be active before the failing frame.
It is included in `npm run test:dawn:sparse-cm12:deficit-support`.

Before the repair this test reproduces the exact supplied halt; after the
repair the full four-second run passes. Raw reproduction evidence is under
`artifacts/half-pool-deficit/exact` (including steps 41–43); the failing donor
and activity masks above were read from those native buffers.


## Forced-finest negative-density receipt

The second supplied configuration adds a whole-tank minimum/maximum cell-size
region of one finest cell. It reproduces `INVALID_CONSERVED_VALUE` at frame 43,
with density `-0.0005950927734375` and gamma `1.0009613037109375`. The native
capture can select a reflected first reporter (192832 instead of 20608); both
cells carry the exact reported values.

At step 42, cell 20608 at `(0.5,40.5,2.5)` is dry, has gamma
`1.001556396484375`, and belongs to the accepted cell worklist. Step 43's gamma
diffusion leaves a receipt of -39 fixed-point units in conditioning plane one.
The next transport gather reads that same plane as forward-deficit density:
`-39 / 65536 = -0.0005950927734375`.

The clear missed the cell because the accepted worklist omitted its page while
the page's active flag and transport execution image still retained it. The
four corner pages `(0,5,0)`, `(7,5,0)`, `(0,5,7)` and `(7,5,7)` all had retirement
intent but no scheduled publication. After step 43, the cell worklist contained
149,504 cells (292 B8 pages), while activity reported 296 active pages.

`validateCandidateResolution` allowed slot-free activation and dynamic-page
retirement, but skipped retirement of fixed-rung authored leaves without a
packed candidate slot. `scheduledBrickActive` nevertheless used their pending
retirement intents when building the shared topology. Another leaf's valid
transaction could consequently publish only half of their lifecycle change.

The repair admits same-rung authored retirement into the ordinary transaction:
it has no candidate field to store. Its face-transfer path publishes a zero
receipt without indexing a nonexistent candidate slot. Scheduled membership
now uses candidate activity only for scheduled leaves, mirroring the existing
scheduled-resolution rule; unscheduled intent cannot change a committed image.
No negative density is clamped to hide the mismatch.

Both receipt fixtures run for 120 steps. The forced-finest regression also
compares accepted cell dispatch size against every active B8 page at each
step, detecting the topology mismatch before stale scratch becomes density.
Native evidence is under `artifacts/half-pool-negative/scratch`.

## Validation status

Both full-scene regressions pass with nonnegative density and less than 0.1%
mass drift, including the per-frame forced-finest worklist/active-page check.
The frontier-election test, mini64 deficit test, and both directory contention
and tombstone-reuse tests also pass. Some combined-run attempts were excluded
by another task's GPU lease; the affected files were rerun after it released.
Six targeted topology/retirement unit checks pass.

The canonical gate was run with its unchanged limits. Its first run exceeded
180 seconds after lane timeouts; five lanes passed before the budget was
exhausted. Type checking still reports errors elsewhere in the checkout;
there are none in this repair's touched files. Broad lint and older structural
source tests also report existing unrelated failures. Focused lint passes for
the new regressions, diagnostic decoder and capture tool.

## Default mini64 UI cadence (third receipt)

The default mini64 receipt reproduces in Dawn with freeze off, without forcing
`waitForTopologyReady()` or dense field readbacks between frames. It halts at
frame 4 / generation 6 / time 0.1666667 s. The first atomic reporter varies,
but donor 179487 has the receipt's exact density, 0.4270046651363373.
Evidence: `artifacts/mini64-ui-deficit/ui-cadence`.

Required CPU support planning begins with asynchronous preflight readbacks.
During that interval `topologyPreparationPending` is still false. Default
`advanceTo` admitted another transport step because its pending-planning guard
only applied to frozen topology. Transport could outrun required backing.
The earlier Dawn harness concealed this by flushing topology after every step.

Admission now also waits for outstanding planning when
`generationPlanningRequired` is true. Fully backed optional planning retains
its existing behavior. The exact exported receipt passes 12 steps in the same
UI-cadence probe after this change (`ui-cadence-fixed`). The mini64 regression
now includes the exported defaults without intermediate topology waits or field
readbacks, plus a unit check for admission during the preflight interval.

The two mini64 Dawn regression variants both pass, as do both planning unit
checks and focused lint. The post-admission canonical gate was rerun unchanged:
five lanes passed, six timed out, and six were not reached before the 180-second
suite budget expired. Log: `/tmp/mini64-admission-canonical-gate.log`.
The broad gate is not green; no timing ceilings or lanes were relaxed.
