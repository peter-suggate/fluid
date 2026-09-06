# Progress owned by the work

`lib/core/work-progress.ts` defines the serializable progress vocabulary. Producers
supply a label, active/waiting/complete/error state, optional counts and unit,
phase names, explanation, generation, and start time. `components/WorkProgress.tsx`
renders those facts in either an inline stage card or a compact background card.
It does not know about mesh headers, resource lanes, or shader names.

The visual grammar is shared:

- Known work uses a real count and progress bar. Counts are local to the current
  phase, not an estimate of total loading time.
- Work without a denominator uses an indeterminate bar and elapsed time when a
  producer provides a start time. There is no fabricated percentage or ETA.
- Waiting/fallback work names its reason and has no animated completion bar.
- Phases highlight only the current phase. They do not infer that earlier phases
  completed, since work can restart or revisit storage allocation.
- Mesh builds identify their generation and source-change reason; brick counts
  measure extraction, while quad demand and storage remain separate diagnostics.
- Readiness appears only after the producer reports completion. A resource whose
  task count reaches its denominator still says it is finalizing until attached.

`ResourcePluginDefinition.progressPhases` declares labels beside each resource
owner, and existing `phaseCopy` supplies explanations. Existing plugins without
phase metadata remain supported using their actual activity label and counters.
The WebGPU platform and SVO presentation definitions demonstrate named phases.
Startup cards and background resource cards use the same renderer.

`surfaceMeshProgress` lives next to the mesh WGSL and interprets mesh counters and
fallback states for the frame panel. Mesh storage expansion is a visible phase,
not a reset to zero. Current-frame ray fallback during preparation is explicitly
explained. No state assumes a stationary scene or camera.

The UI announces changing phase labels politely, without announcing an elapsed
clock every second. Native progress semantics expose counts to assistive tools.
