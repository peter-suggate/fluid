# Lane F (fluid kernels) progress

Design: `docs/plans/nbflip-structural-2026-11-15.md` (§2.4–2.6, §4.2, §5, §7.1). Scratch: `~/code/me/fluid-scratch/lane-f/`.

## M0(c): the p1b "collapse" root-caused (2026-11-15)

**Verdict: p1b never collapsed. The runs that showed ~7.9k particles ran a different scene.** There is no hidden
dependency between the move and `nbSurfaceShift`; moving the shift out of the move is safe for `pBand`.

**Cause.** P2's p1b attribution runs passed `--scene=letters` to `tools/nbflip-attribution.ts`, whose scene
argument is a raw scene id. `getSceneDefinition` (`lib/core/scenes.ts:2684`) falls back to `SCENE_CATALOG[0]` on an
unknown id, without an error, so "letters" ran `water-box-dam-break` (no NB-FLIP profile, balanced defaults). The base
and p1a runs passed `--scene=nbflip-figure-8-letters`. The logs show it: `attr/L120-base.log` and `L120-p1a.log` print
`"scene":"nbflip-figure-8-letters"` (886k / 885k particles, 7972 / 8026 fine tiles), and both p1b logs print
`"scene":"letters"` (8057 / 7983 particles, 80 / 81 fine tiles, 25 ms a step). The "check on is healthy" run used
`tools/nbflip-move-check.ts`, whose `--scene=letters` is a key mapped to `nbflip-figure-8-letters`. So "check on /
check off" was confounded with "right scene / wrong scene"; the check itself changed nothing.

**Reproduction** (check slot, rc 0 each, f72b3ca6 exports, `tools/p1b-probe.ts`; logs in `lane-f/logs/`):

| tree | `--scene` | resolves to | particles at step 119 | fine tiles |
|---|---|---|---|---|
| base f72b3ca6 (unpatched) | `letters` | water-box-dam-break | 8,020 | 84 |
| p1b, check off | `letters` | water-box-dam-break | 7,961 | 94 |
| p1b, check off | `nbflip-figure-8-letters` | nbflip-figure-8-letters | 884,328 | 7,835 |

The unpatched engine gives the same ~8k on the fallback scene, and p1b with the check off is healthy on letters
(884k against the base's 886k in P2's run; NB-FLIP is nondeterministic). With P2's in-situ check (0 bitwise
mismatches over 120 letters steps, about 94% of corrections taken from the pre-pass), p1b is the same computation as
the base.

**Consequences.**
- §2.4's hazard is closed: `pBand` may form `nbSurfaceShift(q, nbSeparation(...))` from the pre-move bank in a
  separate pass. The surface shift samples the depth field at the record's pre-trace `q`, which the move never
  writes, so a pre-pass and the in-move call are the same function of the same inputs (P2's check confirms it
  bitwise). The only fallback cases are a cell-cap merge that moved the sample or a rigid contact that pushed it.
- P2's p1b timing (`L120-p1b.json`) is of the wrong scene and is void; p1b on letters is untimed. It is moot for the
  new engine, and not worth landing on the old one.
- Tooling: `tools/nbflip-attribution.ts` now uses `findSceneDefinition` and fails on an unknown id (AGENTS.md:
  unknown kinds must fail explicitly). `nbflip-gate-bench.ts` and `nbflip-move-check.ts` map keys and were not
  affected.
