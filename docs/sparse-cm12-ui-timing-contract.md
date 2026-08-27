# Sparse CM12 UI timing contract

The SIM panel is an execution diagram, not a conceptual algorithm diagram. A
row's label and tooltip must describe every GPU command between its timestamp
seams, including allocation, worklist construction and buffer copies.

## Current correction

The historical `activity-measurement` row also publishes dirty masks, ages and
seals the activity census, and discovers/synthesizes dynamic sparse-world
frontier pages. It is shown as **Activity census + frontier**.

The historical `resolution-planning` row does substantially more than choose a
rung. It grades to 2:1 closure, activates and retires pages, schedules the
budget, allocates and synthesizes candidate cells, builds the shadow topology
worklists and copies four indirect argument blocks. It is shown as
**Candidate topology build**.

The old `brick-retirement` row does not decide retirement. That decision occurs
inside candidate topology construction; the later row republishes D4 authority
and the post-commit activity mask. It is shown as **Post-commit activity mask**.

## Drift-prevention plan

1. **Dispatch manifest now.** Truth-sensitive rows own a grouped `timedWork`
   manifest beside their UI declaration. The tooltip is generated from it.
   `sparse-cm12-advance-partition.test.ts` extracts the corresponding encoder
   callback and fails when its shader entry points or command-copy count differ.
2. **Extend coverage stage by stage.** Add a manifest whenever a stage is
   touched, then make `timedWork` mandatory once every resident stage is
   covered. Delete freehand timing prose after the migration.
3. **Move the manifest to the command source.** Replace source extraction with
   typed command descriptors consumed by both `encode()` and the stage
   registry. Shader entry-point moves, additions and removals will then be
   compile errors rather than source-test failures.
4. **Acknowledge semantic shader changes.** Generate a transitive WGSL digest
   for each command descriptor. A digest change requires a manifest revision
   and refreshed user-facing work label, even when the entry-point name stays
   the same.
5. **Keep a live closure gate.** The hardware stage probe must continue to prove
   that every emitted seam appears once in the UI, every row cost is the sum of
   its owned seams, and the stage totals close against the measured GPU span.

The immediate test is intentionally scoped to the two misleading hot rows. It
provides protection without forcing an unrelated rewrite of the rest of the
resident encoder; the typed-descriptor migration removes that temporary scope.
