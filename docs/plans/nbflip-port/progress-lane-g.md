# NB-FLIP port, lane G: cross-scene gates (progress)

Brief: docs/plans/nbflip-port-2026-11-15.md §3 "Lane G" and §4.

## 2026-11-15 STARTED (lane G)

HEAD 46a2ba94, tree clean. Lane A profiles NB-FLIP on the check slot at the same time.

## 2026-11-15 MILESTONE G1: box observer and cross-scene bench

Files:
- `tools/nbflip-observables.ts`: the box schema for the dam break, letters and paddleboats
  (totals, energies, (x, z) bins with liquid and highest wet top, floor front, wet floor,
  the dam's back-wall column), the shared WGSL accumulator and its Float64 oracle.
- `tools/nbflip-observe.ts`: NB-FLIP's owner traversal (the seep observer's, h and 4h,
  canonical phi, six-tetrahedron fill), face-mean velocities, and BULK16 regions decoded
  on the host with their (x, y, z) page and face-mean velocity.
- `tools/nbflip-gate-bench.ts`: the seep-bench loop for `--scene=dam|letters|paddleboats`
  (fenced or pipelined, check or timing slot). The seep keeps running through
  `tools/seep-bench.ts --method=nbflip`, unchanged.
- `tests/nbflip-observer.test.ts`: oracle against closed forms (a level pool: total, PE,
  bins, tops, front, column; a dam box; a BULK16 region as 64 4h owners; the letters' drop
  schedule, steps 25i+3), and the WGSL accumulator against the oracle (GPU, passed under
  `fluid-gpu-run.sh check`).

Smoke checks (check slot, HEAD 46a2ba94 lib):
- Dam break, t = 0: 184.00 L, the analytic 80 x 118 x 80 cells at 6.25 mm (184.4 L by the
  0.92 x 128 height); 24 BULK16 regions decoded; front 80 cells (Z = 1); column top
  117.75 cells (eta = 1.00). The front reaches the far wall (128) at step 11.
- `readStats` returns the volume pass of the last step that took one, so without a request a
  sample's nbDrift is one sample old (dam: nbDrift at 45 = observer at 30). The bench now
  requests the pass before each sample step's advance, as tools/probe-reservoir-bulk.ts does.
- The first dam run was invalid: another process's GPU hang (kIOGPUCommandBufferCallbackErrorHang,
  11:19:31) discarded its command buffers, and it then failed a residency closure at step 30.
  The rerun was clean and passed step 60.
