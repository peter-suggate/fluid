# Solver Comparability Contract

No single mapping makes a MAC grid and DFSPH universally equivalent. Every
comparison declares one of three families and reports both configured and
measured cost.

## Resolution mapping

Let the nominal spatial length be `ell`.

```text
Eulerian cell width:       dx = ell
initial particle spacing:  s  = ell
SPH compact support:       R  = 2s (default; exact kernel convention recorded)
initial particle volume:   Vp = s^3
particle mass:             mp = rho0 s^3
```

For liquid volume `V`, first-order counts are:

```text
Eulerian pressure unknowns       approximately V/dx^3
Eulerian velocity unknowns       approximately 3V/dx^3 plus boundary faces
DFSPH particles                  approximately V/s^3
DFSPH vector velocity unknowns   3V/s^3
```

Actual active/allocated counts supersede estimates. Level-set markers, air-band
cells, solid fractions, boundary particles, hash tables, solver work vectors,
and rendering copies are reported separately.

## Comparison families

1. **Equal nominal length:** `dx=s=ell`; best for surface-feature scale, but SPH
   support spans more than one nominal element.
2. **Equal active fluid DOFs:** adjust `dx` or `s` until measured evolving fluid
   unknowns are approximately equal; report pressure/constraint auxiliary state.
3. **Equal compute budget:** choose resolutions whose median step time or time
   per simulated second is within 5% after warm-up on the same hardware.

## Required measurements

Every comparison exports initial fluid volume/mass, active unknown counts,
allocated and peak resident bytes, iteration counts, neighbour pair evaluations,
per-stage CPU/GPU time, accepted step sizes, and simulated seconds per wall
second. Common observables use identical world-space probes and timestamps:

- volume and centre of mass;
- total linear/angular momentum and mechanical energy;
- body position, quaternion, velocity, angular velocity, forces, torques, and
  displaced volume;
- free-surface heights at fixed probes;
- dam-break front position using a frozen threshold definition.

Grid divergence and particle density error are not converted into a synthetic
shared score. They are shown as native constraint errors beside common physical
observables. Long-horizon field differences are summarized statistically because
particle and interface trajectories can diverge chaotically.
