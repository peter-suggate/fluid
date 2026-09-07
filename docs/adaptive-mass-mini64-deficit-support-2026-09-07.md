# Mini64 forward-deficit support repair — 2026-09-07

The supplied `minimal-power-dam-break-64` receipt halted at 0.2 s with
`EMPTY_DEFICIT_STENCIL`, frame 5, `visibleWeight=0`, `deficit=1`. The scene uses
balanced adaptive-mass defaults, 1/30 s steps, and evolving topology. The unused
LoSasso overrides in the receipt do not change adaptive-mass method values.

The repaired scene completes 12 steps (0.4 s) without a support fault. This is an
allocation and support-completeness repair. It does **not** establish physical
correctness: substantial over-compression remains, described below.

## Mechanisms and changes

### Hash contention postponed a required receiver

The exact scene/configuration reproduced the supplied failure on the current
checkout at the same frame/time. A CPU generation rebuild changed the stable
owner numbering: this capture reports owner 58917, generation 5. That donor is
at `(53.5, 0.5, 39.5)` in finest-cell coordinates. Its prior density was
`0.00021853268845` and collocated velocity approximately
`(126.7203, -15.6263, 36.3762)` finest cells/s.

At the end of step 5, its source page `(6,0,4)` already requested the diagonal
receiver `(7,0,5)`. That page was absent. Its hash starts at slot 435 of the
2048-entry world directory; another requested page `(6,1,4)`, allocated during
that same frame, starts at slot 435 too.

`cm12WorldAllocateExact` stopped when it encountered any in-progress hash
reservation. This prevented duplicate sources from allocating one coordinate
more than once, but it also deferred a distinct colliding coordinate until the
next topology epoch. The next physics step needed that receiver immediately.

The frontier now elects one source for each absent target using the immutable
activity masks and SolidWorld reachability. It checks actual macro origins so a
covering macro leaf cannot impersonate a different source coordinate. Inactive
backed leaves cannot win. A unique-request insertion can then continue probing
past another reservation without reading its unpublished key. Live interaction
allocation already assigns one invocation per coordinate and uses that same
insertion mode. The duplicate-capable allocator remains available for callers
without this uniqueness proof.

CAS retries of one slot do not consume the directory's slot-search budget.
Leaf-capacity exhaustion still reports the existing capacity fault and creates
no leaf. No transport fallback or extra allocation dispatch was introduced.

### A dilute wall-tangent characteristic lost its open route

Fixing the collision exposed a later support fault at step 9 / 0.3 s. The donor
was at `(32.5,9.5,63.5)`, density `0.00132138655`, in page `(4,1,7)`. Its
collocated velocity was `(47.27,55.16,139.98)` finest cells/s. The following
velocity extension produced effective transport velocity
`(71.93,299.94,24.61)` there. A CPU replay of the native frozen fine-grid velocity
cache reaches approximately `(34.37,16.60,63.5)`, inside absent page `(4,2,7)`.

The activity mask already included the predicted `(0,+1,+1)` direction. The
solid z wall rejected that diagonal route. The mask did not include its open
`(0,+1,0)` component, so no upper tangential page was requested. In addition,
the allocation gate required the coarser `occupied` flag even though the census
already produces explicit swept support for **every nonzero mass donor**.
That discarded dilute donors' requests.

The existing census now emits the complete page box between the source and
its predicted endpoint, including face/edge components of diagonal travel.
SolidWorld can reject blocked components while retaining the open ones. The
explicit support mask authorizes allocation directly; the occupied/surface
classifier continues to govern its separate resolution decisions. The
prediction reach, velocity extension, characteristic integration and numerical
pass count are unchanged. No density cutoff was reduced or used to discard
mass.

## Validation

Run the focused regression with:

```sh
npm run test:dawn:sparse-cm12:deficit-support
```

**Result: 4 passed, 0 failed in 60.9 s.** The ordinary mini64 regression
completed its 12-step assertions; the process and GPU lease exited cleanly.
The run log is `/tmp/mini64-deficit-support-gate.log`.
This is the recorded four-test run; a separate half-pool repair subsequently
added its longer receipt regression to the same package script.

It covers:

- 64 distinct keys sharing one hash bucket at the directory's maximum supported
  50% table load, all allocated in one dispatch; six tombstone reuse cycles;
  lookup completeness and explicit full-capacity rejection. A control using
  the original duplicate-capable insertion publishes only one of those keys
  in one epoch, reproducing the mechanism.
- Frontier source election with signed coordinates, diagonal requests, a macro
  origin, blocked routes, a dilute source without the occupied flag, and an
  inactive backed source.
- All 27 signed sweep directions; reflection and axis-permutation invariance;
  the blocked z-diagonal case retaining its open y request.
- The ordinary balanced mini64 scene through 12 steps, including the original
  colliding receiver pages by step 5, finite fields and mass preservation.

The exact receipt capture also completes all 12 steps. Page `(7,0,5)` and the
competing `(6,1,4)` are resident by step 5. The second missing receiver `(4,2,7)`
is resident by step 8. Initial mass is `94208.0000610` finest-cell equivalents;
final mass is `94207.3887017`, a drift of **−0.000649%**.

Artifacts:

- `artifacts/mini64-deficit/before`: exact-scene reproduction before the repair;
  log `/tmp/mini64-deficit-repro.log` includes the original current-checkout fault.
- `artifacts/mini64-deficit/unique-allocation`: isolates the hash repair and
  captures the later wall-tangent failure, including native state/activity.
- `artifacts/mini64-deficit/swept-box`: complete 0.4 s capture after both support
  repairs, with configuration, native state, activity and trajectory receipts.
- `artifacts/mini64-deficit/swept-box/native-density-summary.json`: root task
  audit of accepted-cell density peaks and oversaturation, with cell positions.
- `tools/probe-mini64-deficit-dawn.ts`: receipt-driven capture utility; pass
  `MINI64_DEFICIT_RECEIPT` to the supplied JSON file and set `MINI64_DEFICIT_OUTPUT`
  for an isolated output directory.

Targeted lint passes. Repository type checking still reports existing errors;
there are no errors in this repair's directory, frontier, probe or regression
files. The final canonical sparse-CM12 Dawn gate was run after integration with
dynamic freeze and signed-world presentation changes. It remains red on the
density-symmetry assertion, page-budget/hydrostatic timeouts and performance;
six lanes pass, including four-second mini32 correctness, and six tail lanes
remain unrun within the 180-second budget. Its receipt is
`artifacts/minidam32-frozen/dynamic-freeze-final-gate.json`. No ceiling or budget
was raised for this repair.

## Unresolved physics

The successful capture reaches density **10.836895** in an accepted cell at
step 12. The root task's native accepted-cell audit measures positive
oversaturation of `10785.8086` finest-cell equivalents, approximately **11.45%**
of initial mass, with 20,773 cells above `rho=1.001`. The peak cell belongs to
dynamic leaf `(1,0,7)`.

This problem predates the newly allocated frontier: at step 4, an accepted
initial-template cell at `(34.5,5.5,19.5)` already reaches density `1.978818`.
Allocation completeness allows this trajectory to be examined; it does not
repair over-compression or prove realistic dam-front motion. Those remain part
of the main physics/correctness investigation.
