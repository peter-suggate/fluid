# Pressure bounds and safeguarded recovery: implemented in 2D, then 3D

Both backends now propagate pressure lower bounds through every level, including
shifted bounds in Full-Cycles and correction bounds in V-cycles. The arbitrary
three-transfer cutoff has been removed from the shared policy and generated Rust
contract. The ordinary 3D pressure implementation shares this solver with the
uniform density method, so it receives the same protection.

Each solve measures the initial projected residual and retains a dedicated best
pressure field. After a complete cycle, it rejects a non-finite or worsening
candidate, restores the best field, and stops multigrid for that step. This guard
remains enabled when the residual tolerance is zero. Residual evaluation also
checks pressure-bound violations and non-finite inactive pressure values.

After rejection, recovery runs up to eight batches of eight projected fine-grid
red-black sweeps. Its finite working iterate can temporarily worsen the infinity
norm; only an iterate at least as good as the retained best can be published.
Non-finite working iterates restart from the best. Recovery stops early only when
it reaches both the configured absolute tolerance and a tenfold reduction of the
initial residual. Otherwise it retains the best and explicitly reports exhausted
recovery. There is no silent claim that an exhausted solve converged.

The additional relative recovery target matters for rigid coupling: stopping
recovery after eight sweeps at the default absolute tolerance of 10 under-solved
the buoyancy fixture. With the relative target, the unchanged scalar/SIMD live-tool
suite passes; the submerged disk rises from 0.3 to 0.3214018 m in 30 frames. Some
embedded-body solves still exhaust recovery or hit the coarse iteration cap;
this is the motivation for the subsequent active-set work, not a claim that all
coarse systems are now efficient or tightly converged.

GPU decisions and restoration stay on the GPU. A dedicated finest pressure texture
cannot alias Full-Cycle backup or V-cycle scratch. Existing diagnostic words retain
their offsets; appended words report accepted and initial residuals, rejected
cycles, recovery sweeps and exhaustion. Recovery dispatches are GPU-gated when
unneeded. The implementation adds one finest scalar texture, per-cycle acceptance
passes, and a bounded recovery schedule; it does not add a CPU readback to advance.
This work establishes correctness gates, not a matched-residual performance claim.

Validation:

- Four Rust guard tests pass: all six bound transfers, finite/non-finite rollback,
  recovery and exhaustion, and inactive-NaN/bound-violation detection.
- The original failing pressure matrix is preserved in
  `frame-104-pressure.json.gz`. `uniform_geometric_pressure` replays it directly.
  Two Full-Cycles give residual **0.359018**, maximum pressure below 1e6, and no
  rejection or fallback; the original result was **870531.94**. The fixture uses
  tolerance zero to execute both cycles, so its `converged` flag remains false by
  the existing disabled-early-exit convention.
- `test:uniform-pressure:2d` passes the frozen matrix and 180-frame authored Figure 9
  run, with exact native/scalar/SIMD fields and receipts. Peak speed **43.1819 m/s**;
  final speed **7.3457 m/s**, phi area **4049.2083**, V **3839.9925**. No rejected
  cycles in this run. See `stability-2d.json`.
- The full **128×128×64** 3D scene passes 180 frames / six seconds. Peak speed
  **70.3399 m/s**, final speed **8.2145 m/s**, final accepted residual **2.48566**.
  No rejected cycles, surface survives, and mass checks pass. See `stability-3d.json`.
  The solver's quantized mass telemetry reports -1.26e-5 relative drift; a separate
  full-volume texture sum measured -1.48e-6. These are different measurements.
- GPU fault injection passes for huge finite pressure, runtime NaN, runtime
  infinity, and corruption after an already accepted cycle. Tests check bounded
  projected velocity, residual acceptance, per-step counter reset, and explicit
  exhaustion with tolerance disabled.
- The final combined Dawn run passes **24/24** checks; see `dawn-validation.txt`.
  Existing hydrostatic, transport, pressure-parity/reset, separating-domain and
  embedded-wall, Figure 8/12, and sharpening-work-map checks pass.
- All **33** GPU-2D/native/scalar/SIMD one-step parity checkpoints across **11** scenes
  pass without changing their bounds. See `parity.json`.
- Scalar/SIMD Wasm artifacts were rebuilt and validated. The existing live-tool
  acceptance suite passes. Targeted lint and `git diff --check` pass.

Broad repository checks still encounter unrelated existing errors: the all-target
Rust test command fails to compile `world3d_minimal_power.rs` because its
`WorldOptions` literals omit `adaptive_sdf`; targeted library tests pass. Typecheck
reports existing sparse test/tool errors (and the concurrently edited lab state
briefly introduced an unrelated fixture type mismatch). No pressure-code errors
were reported. Sparse CM12 implementation/topology/publication was not changed, so
its post-refactor gate is outside this change's scope. GPU runs were serialized
through the repository lease, including waiting for another task's overlay test.

The pressure fix does not repair phi/V drift: the six-second 2D phi area is still
about 5.45% above conserved volume, and the 3D surface-volume discrepancy is larger.
The [active-set multigrid plan](../../plans/uniform-active-set-multigrid.md) covers
constraint-compatible transfers, a frozen-system reference corpus, staged 2D/3D
implementation and acceptance criteria. It is planned, not implemented here.

Reproduce:

```sh
npm run build:physics-wasm:single
cargo test --manifest-path rust/Cargo.toml -p fluid-core --lib safety_tests
npm run test:uniform-pressure:2d
npm run test:uniform-lab:scenes
npm run test:dawn:uniform-pressure
npm run probe:uniform-geometric:parity -- --wasm --verify
```

Run Dawn commands serially and without an active browser GPU simulation.
