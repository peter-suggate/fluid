# Small-fluid work reduction: analysis and proof

Status: the analysis below established the work-count and dependency proof. The first candidate is now implemented; see [implementation and measurements](cm12-vex-rebuild-work-reduction-2026-09-11.md). The further fluid-sized selection remains analysis only.

The target is Figure 7, sphere radius 0.1 m, B8/P8, scene timestep, pressure tolerance 0.194. The current production CPU initialization was rerun. It still produces 36 accepted B8 leaves (four containing liquid), 18,432 accepted cells, 64 nonzero-density cells and 32 velocity-extension seeds. The existing 432 reserved growth pages bring the direct VEX domain to 3,744 packets; only 288 packets are accepted at reset.

## 1. First candidate: retire invalid masks once, then use the existing accepted list

The existing VEX schedule already compiles an accepted-packet list. However, `sealSparseCM12VelocityExtensionSchedule` forbids its use whenever topology generation or slot changes. The host then sends **initialization and all eight sweeps through the same direct dispatch arguments**. This scans every reserved packet nine times.

For an invalid packet, initialization clears its two A-bank mask words and returns. Each sweep loads its invalid TEI descriptor, clears two words of the alternating output mask bank, publishes the frame receipt when applicable, and returns. These packets have no accepted cell arithmetic. The repeated clears are the sole reason given for retaining a direct rebuild frame.

Keep the direct initializer on rebuild frames. Have it also clear the B bank for invalid packets. Then run all eight sweeps on the **already compiled accepted-packet list**. Stable frames retain their current accepted-list initialization and sweeps; dense images retain the existing direct-dispatch occupancy policy.

### Exact reset-sized work counts

| Work | Current rebuild frame | Proposed rebuild frame |
|---|---:|---:|
| Initialization workgroups | 3,744 | 3,744 |
| Eight sweep workgroups | 29,952 | 2,304 |
| Total workgroups | **33,696** | **6,048** |
| Total 64-lane invocations | **2,156,544** | **387,072** |
| Invalid-packet mask stores, including initialization | 62,208 | 13,824 |

This removes **27,648 workgroups / 1,769,472 lane invocations (82.05%)** per reset-sized rebuild frame. It also removes **48,384 32-bit mask stores**. The accepted numerical work is identical. Invalid invocations already return early, so these percentages are not predictions of elapsed-time savings.

### Dependency and stale-state proof

1. The accepted list comes from the same current TEI descriptors used to recognize valid packets in direct execution. Keep its existing generation/slot checks.
2. Initialize every accepted cell exactly as before: liquid seeds receive their source velocity and depth zero; other accepted cells receive zero effective velocity and invalid depth. Keep the direct initializer's coverage of retired packet addresses.
3. Clear both mask banks for invalid packets. After sweep one, both the old and proposed schedules therefore have identical A and B banks, including arbitrary stale bits left by prior topology. Every subsequent invalid-packet write in the old schedule merely writes zero again.
4. Accepted-cell recurrence reads neighbours with accepted depth **strictly less than the current sweep depth**. Cells newly written in this sweep cannot affect other cells in that sweep. Moving accepted workgroups into the existing compact order therefore preserves the recurrence and floating-point operation order within each cell.
5. Preserve singleton frame-receipt publication, including the empty-list case (one guarded workgroup), the partial-packet lane guards, and the existing dense-image mode.

The CPU proof starts the old B bank with adversarial stale bits and verifies equality of both banks after every sweep. It allows arbitrary accepted output masks, so this proof is not dependent on the scene's initially zero velocity.

### Added work must remain bounded and explicit

No new work-selection traversal is needed: the accepted list already exists. No extra simulation pass is needed. Initialization needs two extra stores per invalid packet, which replace sixteen later stores per invalid packet.

Initialization and sweeps need separate dispatch mode/argument handling. The existing 24-byte indirect buffer has two records: transport at offset 0 and VEX at 12. Transport's arguments are not published until VEX finishes, so its first record can carry VEX initialization arguments temporarily. The existing schedule-seal pass can publish both argument triples, with a 24-byte copy replacing the current 12-byte copy. The schedule metadata would need three additional words (12 bytes before allocation alignment); the indirect buffer need not grow. Initialization derives its direct/compact mode from the existing rebuild and occupancy flags. This is a proposed integration plan, not implemented code.

Source anchors:

- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts`: `beginSparseCM12VelocityExtensionSchedule`, `sealSparseCM12VelocityExtensionSchedule`, `initializeVelocityExtensionPackets`, `advanceVelocityExtensionPackets`.
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`: transport-velocity-extension encoder and allocation of `transportPacketIndirectArguments`.
- `lib/methods/adaptive-mass/sparse-cm12-velocity-extension.ts`: eight-word schedule and dispatch-shape layout.

### This is recurring work, not just bootstrap

The existing final Dawn receipt records **20 distinct input-topology generations over 32 advances**. Each transition necessarily rebuilds this schedule; slot-only changes could add further rebuilds. This is different from the narrower pressure-topology change attribution, which must not be used as the VEX rebuild count.

The terminal VEX receipt independently reports: frame 32, topology generation 20, rebuild true, compact false, **3,744 dispatched packets versus 232 accepted packets**, 2,328 valid cells, 3,664 empty direct-domain packets, and zero faults. At that frame's sizes the same change would remove **28,096 sweep workgroups**. This receipt supports the work-domain diagnosis; it does not provide per-frame accepted-packet counts for all 32 advances.

## 2. Further fluid-sized reduction: proven support bound, selection cost unresolved

Even the accepted list includes much unnecessary air. For this exact B8 reset, walk the production composite grid's six-neighbour physical graph from the 32 rho > 0.5 seeds. The valid-cell counts at depths 0 through 8 are:

`32, 80, 160, 280, 448, 672, 960, 1,320, 1,760`.

All cells reachable within eight sweeps fit inside **56 of the 288 accepted 4³ packets**. A cheaper conservative construction, retaining only the four seed-packet boxes and their possible eight-step reach, selects **112 packets**. Both selections contain every actual dependency.

| Eight sweeps, excluding retained initialization | All accepted packets | Exact reachable packets | Conservative packet boxes |
|---|---:|---:|---:|
| Packets per sweep | 288 | 56 | 112 |
| Workgroups over eight sweeps | 2,304 | 448 | 896 |
| Lane invocations | 147,456 | 28,672 | 57,344 |
| Actual neighbour-depth reads in the reset graph model | 811,872 | 146,272 | 308,064 |

The CPU exhaustive and restricted recurrences agree on **every valid cell at every intermediate depth**, not merely the final count. Initialization still visits all accepted cells to erase stale effective velocities/depths. Skipped accepted packets also require their old B mask cleared. No extension depth, interpolation support or fluid seed predicate is weakened.

This demonstrates **61.1–80.6% fewer sweep invocations** are geometrically possible at reset. It does **not** yet prove net savings after constructing the selection. Exact graph reach is not free. A production selector must handle mixed rungs, sparse-world boundaries, newly inserted liquid and moving solids, and its construction cost must be included. Existing masks do not already supply this exact worklist. Therefore this is the second candidate, not the first implementation.

## 3. Why the other large stages are not yet proven removable

- **Face preparation:** only 5,940 of 58,368 reset rows reach tracing, but the other rows currently clear unsupported destination velocities. Simply dropping those rows can retain stale velocity. A smaller execution set needs a proven current/previous support union and retirement coverage. This proof is outstanding.
- **Capacity repair:** eight rounds cause 442,368 accepted-cell visits at reset-sized topology. Initial density below capacity does not prove absence of excess after transport, diffusion and sharpening. The code already retains a rejected early-exit experiment; adding another per-lane gate is not evidence of reduced dispatch work.
- **Activity and topology:** sparse support and future fluid arrival make some apparently dry neighbours necessary. Capacity-sized scans are suspects, but each omitted visit must be checked against growth, refinement, retirement and live insertion. No reduction percentage is claimed yet.

## Reproduction and evidence

```sh
node --import tsx tools/census-cm12-figure-7-radius-01.ts
python3 tools/analyze-cm12-small-fluid-work.py
```

- Fresh production reset: `artifacts/cm12-figure-7-radius-01/cpu-census.json`.
- Count and recurrence proof: `artifacts/cm12-figure-7-radius-01/small-fluid-work-proof.json`.
- Existing isolated Dawn run: `artifacts/cm12-figure-7-radius-01/pressure-row-load-final.json`.

The browser was left untouched and no new Dawn process was started. No GPU speedup is claimed. The first candidate now has enough structural evidence to implement narrowly and then validate stale-mask retirement, partial/rerung packets, empty scenes, live edits, full regression behavior and total GPU time.
