# Relayout lane: progress

Goal: make relayout cost scale with the tiles that change (paddleboats, target −1.5 to −2.5 ms).

## 2026-11-03: knockouts, no landing

All runs are paddleboats with `--bodies --steps=90`, kernels taken from step 60, one run each. They ran on a HEAD 09b89e0d export with env-gated knockouts. The census changes the layout on 90 of 90 frames. The remap list holds 158–272 jobs a frame.

Base: kernel sum 46.1 ms. Remap passes 1.09 ms, extension passes 3.76 ms.

### Remap (already changed tiles only)

The remap and the regional re-extension the brief asked for are already in place:
- **Remap:** `markListed` builds its list from the generation's dilated changed tiles, so the remap already touches only changed tiles and their halo. There is no lattice scan to remove.
- **Re-extension:** the head extension runs before `relayout.encode`, and it is already regional when only bodies changed (`bodyExtension && same`). The relayout itself does not trigger a full re-extension.

| Knockout | `remapFaces{twinned}`, ms (2 launches) |
|---|---|
| base | 0.706 |
| V evaluation replaced by 1 (RK_V1) | 0.353 |
| `remapBeside` and `remapSlope` skipped (RK_NOSLOPE) | 0.404 |
| released test skipped (RK_NOREL) | no change |
| extension-field launch skipped (RK_NOEXTREMAP) | 0.357 |

Other remap kernels:
- `planeSums`: 0.09–0.16 ms. This is under the 0.3 ms bar, so it gets no parallel combine.
- `remapCells` 0.09, `copyFaces` 0.09, `markListed` 0.03 ms.

Reusing V from the extension launch in the velocity launch would save at most about 0.17 ms. That needs a per-job V buffer. Dropped: under the bar.

### Head extension (before the relayout)

| Knockout | Effect on kernel sum |
|---|---|
| Head extension removed (RK_NOHEADEXT) | −1.78 ms |
| Forced full rather than regional (RK_FULLHEAD) | +0.5 ms |

Breakdown of the head extension:
- hierarchy 0.59 ms
- publish 0.36 ms
- regional seeds and sweeps 0.63 ms
- marks 0.08 ms

REGHIER is an inexact upper bound. It runs the hierarchy and publish only over blocks holding a dirty or halo region. It saved −0.27 ms, because dirty and halo regions already cover most held blocks. A regional hierarchy is therefore dropped.

### Exact kernel change tried: staged 4h/8h prolongation

The change: a prolongation job of an edge-4 or edge-8 level stages its 64–72 coarser taps (state and both source bounds) in workgroup memory. Each lane's eight taps then read from there.

Correctness: on `tools/probe-uniform-extension-pyramid.ts`, every output line matched HEAD exactly (the timing column was ignored).

Timing: prolong went from 0.641 to 0.545 ms, a saving of 0.10 ms. Dropped: under the bar. The kernel is kept at `scratchpad/relayout/staged-prolong.wgsl.ts` in case a later change needs the extra 0.1 ms.

### Pyramid probe

`tools/probe-uniform-extension-pyramid.ts` no longer compiled. Its far-lookup module lacked `umSolidIsBox`, `umBulkSlotsOn` and the phase that `ueFarOpen` reads. It now has a stub for each.

It runs, but it reports 26 860 differences from the host twin, both at HEAD and with the staged prolongation. The twin has drifted from the kernels. This predates this lane and is left for an owner.

### Where the relayout milliseconds are

The share of relayout cost that is proportional to unchanged tiles is small:
- regional hierarchy: about 0.27 ms
- V reuse: about 0.17 ms
- staged prolongation: about 0.10 ms

Even taken together, these come well short of the target.

The larger per-relayout costs sit in other lanes' files:
- census (`classify`, `classifyCoarse`, `decide`): 1.5 ms. It already replays deep COLD tiles.
- post-relayout authority `encode`: 1.37 ms. It is already stamp-regional and lives in the pressure authority.
- geometry (`geometryRegional`, `geometryChanged`): 1.1 ms.
- solid record (`buildRegions`): 0.9 ms.
