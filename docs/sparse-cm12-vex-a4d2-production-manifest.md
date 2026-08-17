# VEX1 + A4D2 production integration manifest

The executable manifest is
`sparse-cm12-vex-activity-production.ts`; the fail-closed gate is
`tools/check-sparse-cm12-vex-activity-production.ts`. They are standalone and
do not modify the resident. The checker composes the actual FCA1, FPL1, FPP1,
SCA1/SAW1, A4D2, and VEX1 layout factories rather than duplicating their ABI.

## Arena composition

The accepted activity prefix ends after dirty scheduling, ACT1/PCM, and other
resident-owned predecessors. Production appends, each at its factory-required
alignment:

```text
activityPrefix -> FPL1 -> FPP1 -> SCA1 option-B receipts -> A4D2 -> VEX1
```

FCA1 begins at the 256-byte-aligned end of the complete mutable topology/page
arena. SAW1 remains a dedicated planner buffer. VEX1's persistent
`vec4f[cellCapacity]` accepted-velocity bank begins at the aligned end of the
complete state allocation, including the pressure journal. Candidate pressure
coefficients or other transient state may never be used as this base.

The production indirect snapshot buffer is 87 u32 words:

| Words | Owner | Contract |
|---:|---|---|
| 0..41 | FCA1 | fourteen fixed triplets |
| 42..59 | FPL1 | six stage triplets |
| 60..62 | FPP1 | presentation page packet |
| 63..71 | SAW1 | mass, gamma, and surface packets |
| 72..83 | A4D2 | classify, scan, rebuild, and census packets |
| 84..86 | VEX1 | serial root/frontier/blast triplet |

The VEX triplet may be reused only across command-order barriers: the storage
header is copied after the preceding indirect consumer has completed. No two
VEX packets may be in flight against that range.

## Producer and receipt contract

All six VEX root origins are mandatory: construction bootstrap, final
projected/collocated velocity, liquid classification/density authority,
topology or extrapolation-edge ownership, injection, and moving solids.
Roots publish FPL stage 0 direct evidence. Positive recurrence depth publishes
closure evidence. Face preparation publishes executed/skipped only after the
eight-edge recurrence has committed.

A4D2 consumes a GPU-sealed, sorted local brick packet. Its 14 required hooks
provide candidate generation/list authority, topology signatures, exact tile
triggers and summaries, FPL root publication, and exact brick/census output.
An A4D2 trigger is provenance, not execution: the accepted activity receipt is
published only after local rebuild, local census reduction, and commit.

SAW option-B owns FPL stages 1 through 3. PCF owns stage 4. FPP owns stage 5.
Every stage must publish executed or skipped for its accepted generation;
unknown/missing coverage faults rather than appearing clean in the Grid
selector.

## Coalescing and lifecycle boundaries

Only these same-authority, contiguous groups may coalesce:

- VEX seed and eight-edge recurrence;
- SAW candidate copy and plan; and
- A4D2 trigger gather and deterministic classification/compaction.

FCA seal, VEX recurrence acceptance, each scalar execution, A4D2 compaction,
A4D2 rebuild/census acceptance, FPL seal, and FPP transaction commit are hard
boundaries. Coalescing cannot cross one of them. Generation/phase transitions
are monotonic per authority and all dependency edges must point backward in
the declared schedule.

The legacy full eight-sweep extension, full activity measurement, and scalar
full path are construction-only QA oracles. They are never runtime- or
output-selectable and are not recovery paths. The checker deliberately mutates
the manifest to prove it rejects arena overlap, missing roots, dependency or
phase errors, global accepted-cell/row dispatch, and selectable QA paths.
