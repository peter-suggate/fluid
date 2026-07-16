# Tall-cell VOF → level-set migration

Status: implemented on 2026-07-16; short native-WebGPU validation complete.

This record supersedes the phased handoff plan. The restricted tall-cell
method now uses one signed-distance field for surface tracking,
classification, pressure ghosts, remeshing, rendering, rigid occupancy, and
diagnostics. The uniform solver remains VOF and the quadtree solver is
unchanged.

## Paper review and the advection decision

The implementation follows *Tall Cells for Liquid Simulation* rather than
applying one transport scheme to every field:

- Velocity uses the paper's bounded modified MacCormack method by default,
  with semi-Lagrangian velocity advection still available as a diagnostic
  option.
- Phi always uses RK2 backtraced semi-Lagrangian advection. Section 7 of the
  tall-cells paper explicitly reports that MacCormack level-set advection
  produced noisy surfaces even when the corrector was treated carefully near
  the interface. Bounded MacCormack is therefore **not** more appropriate for
  phi here; its reduced numerical diffusion does not outweigh that observed
  interface noise.
- Phi is clamped beside the interface every step and reinitialized every ten
  outer steps, matching the paper's cadence. Reinitialization freezes samples
  adjacent to a sign change, operates only within three cells, changes a
  sample by at most one cell per sweep, and clamps the stored band to five
  cells.

*Mass-Conserving Eulerian Liquid Simulation* was also reviewed. Its primary
state is a surface-density/VOF-like quantity, and its Section 3.7 correction
is a local density-excess divergence. That is not directly portable to a pure
level set. In particular, the global controller below should not be described
as an implementation of Section 3.7; it is a separate feedback controller
that borrows only the general idea of correcting volume through divergence.
The paper's density sharpening remains on the uniform VOF solver only.

## Implemented representation

- `initialPhi` is sampled in metres at the paper's Equation 4 packed point
  positions, negative in liquid, and clamped to ±5 times the smallest cell
  spacing.
- A tall cell stores independent bottom and top phi samples. Virtual cubic
  samples inside it use Equation 5 linear interpolation; regular-band samples
  use the direct packed lookup.
- Cubic sampling is trilinear over that reconstructed field.
- Liquid classification is `phi <= 0`. Rendering and test readback convert
  phi to a one-cell transition occupancy only when a scalar occupancy field
  is needed.

The old restricted-path conservative flux transport, reconstructed transport
texture, density-sharpening passes, deposit buffer, and VOF representability
floors have been removed. The public texture property retains its historical
`volumeTexture` name for compatibility, but its restricted tall-cell contents
are phi, as advertised by `info.surfaceField === "levelset"`.

## Solver order

For each submitted step the restricted solver performs:

1. Extrapolate liquid velocity into the air band using the hierarchy.
2. Clamp phi; on every tenth outer step run two protected reinitialization
   sweeps.
3. Advect velocity with bounded MacCormack (or the selected SL diagnostic)
   and advect phi with semi-Lagrangian RK2 tracing.
4. Plan and smooth the surface band, then remap when due.
5. Couple rigid bodies.
6. Build the phi-classified pressure right-hand side, solve with restricted
   multigrid, and project velocity.
7. Reduce reconstructed volume and stability diagnostics.

The finest multigrid level consumes the solver's phi texture directly.
Ghost-fluid fractions use `|phi_liquid| / (|phi_liquid| + |phi_air|)` with
the existing 0.05 floor.

## Remeshing

The planner scans the full virtual column for phi sign changes and applies
the Section 8 liquid/air halo constraints. Equation 10 is strict: adjacent
band bases may differ by at most `maximumNeighborDelta`, with no water-volume
or wet-top representability exceptions.

Tall phi endpoints and velocity endpoints are transferred by least-squares
fits through the old virtual samples covered by the new tall cell. Band
samples use trilinear sampling. If the mandatory minimum two-sample tall cell
would contain a zero crossing at the floor, both endpoint signs are collapsed
to the fitted top sign; this deliberately removes an unrepresentable shallow
film rather than violating the paper's “no interface inside a tall cell”
invariant.

## Volume control

The layout records `referenceLiquidVolume_cells`, increased analytically by
configured inflow. Diagnostics reconstruct a fine-grid occupancy from phi:

```text
alpha(phi) = clamp(0.5 - phi / h, 0, 1)
```

When diagnostic readback completes, the next step uses

```text
c = clamp(0.5 * (V_ref - V_phi) / (N_interface * (1/30 s)), -1/s, +1/s)
```

in cells with `|phi| < 1.5 h`; pressure RHS construction subtracts this
correction from divergence. The method's `volumeControl` parameter and
`FLUID_VOLUME_CONTROL=0|1` smoke override provide an A/B switch. This is a
slow global controller, not exact conservative transport and not the local
density correction from the mass-conserving paper.

## Validation and acceptance

Completed checks:

- `npm run test:unit`: 150 tests, 149 passed, one native-WebGPU test skipped
  when no module is configured.
- Native restricted assignment test at 0.224 s: strict Equation 10 violations
  0, dry tall-top under wet band 0, interface-outside-band columns 0.
- Native settled-tank smoke at 0.02 s: no validation errors or non-finite
  values, one liquid component, no tall/band sign gap, reconstructed drift
  0.61%, and all smoke invariants passed.

The following longer-running acceptance runs remain useful before treating
the migration as a performance/stability baseline rather than a functional
implementation:

| Run | Gate |
| --- | --- |
| settled-tank and deep-water | zero stability flags; reconstructed volume error ≤1%; equilibrium liquid speed ≤0.05 m/s after 0.5 s |
| dam-break-boxes | strict Equation 10; no dry-tall/wet-band gap; volume error ≤1% |
| dam-break-ui, 5 s, control on | no non-finite values; volume error ≤1% at every sample and ≤0.3% at 5 s |
| dam-break-ui, control off | record free drift; do not apply a conservation gate |
| active/soak regressions | projection-energy, CFL, connected-component, IoU, and decay envelopes re-baselined from three level-set runs |
| async browser construction | all tall and multigrid pipelines compile and a rendered step completes |

Volume gates for the restricted method must use `referenceLiquidVolume_cells`
and reconstructed occupancy. Raw phi minima/maxima are not density bounds.
The strict neighbor-delta gate has no representability excuses.

## Known limitations

- The volume controller is updated by asynchronous diagnostic readback, so it
  is one or more submitted frames behind under load.
- The one-cell occupancy reconstruction has an initial discretization offset
  relative to the analytic VOF reference; the controller removes it slowly.
- Level sets intentionally lose sub-grid spray and the shallow films that
  cannot be represented outside the regular band.
- The historical `volumeTexture`/`volumeA` internal names remain and should be
  renamed to `phiTexture`/`phiA` in a future non-functional cleanup.
