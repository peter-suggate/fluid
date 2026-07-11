# Stage 3 — Rigid Bodies Without Fluid

Status: implementation gate  
Scope owner: `RigidBodySystem`

Stage 3 introduces physical rigid bodies but no fluid force. Bodies may pass
through the Stage 2 presentation water field; that field is not solver state.

## Governing equations

For each body with mass `m [kg]`, centre `x [m]`, linear velocity `v [m/s]`,
world angular momentum `L [kg m²/s]`, angular velocity `omega [rad/s]`, unit
quaternion `q`, and world inertia `I [kg m²]`:

```text
m dv/dt = m g + F_contact
dx/dt = v
dL/dt = tau_contact
omega = I_world^-1 L
dq/dt = 0.5 (0, omega) q
I_world = R(q) I_body R(q)^T
```

Velocity-first symplectic Euler advances translation. Angular momentum is
advanced before quaternion orientation; the quaternion is normalized after
every step and its pre-normalization norm error is recorded.

At a contact with normal `n`, relative contact speed `v_rel`, and lever arms
`rA`, `rB`, the normal impulse is:

```text
j_n = -(1+e) (v_rel·n) /
      [mA^-1 + mB^-1 + n·((IA^-1(rA×n))×rA + (IB^-1(rB×n))×rB)]
```

The tangential impulse opposes contact slip and is clamped by
`|j_t| <= mu j_n`. Equal and opposite impulses update linear and angular
momentum. Low-speed contacts use zero restitution to avoid resting jitter.

## Primitive mass properties

- sphere: analytic solid-sphere volume and inertia;
- box: analytic cuboid volume and diagonal inertia;
- cylinder: analytic solid-cylinder volume and diagonal inertia, local `Y` axis;
- capsule: cylinder plus two hemispheres, including parallel-axis terms.

Density is authoritative; mass and inertia are derived. SI units are explicit.

## Boundary and collision choices

The rectangular container uses six analytic inward-facing planes (five for an
open top). Primitive support mappings determine wall contact points. Container
contacts are therefore shape-aware for every supported primitive.

Sphere–sphere body contacts are exact. General primitive-to-primitive contacts
use a conservative bounding-sphere proxy in this first verified increment.
This approximation is deliberately visible in the UI and evidence ledger; it
can cause early contacts for elongated boxes, capsules, and cylinders. Replacing
it with GJK/EPA or shape-pair narrow phases is a later Stage 3 refinement, not a
fluid-coupling concern.

## Tests defined before implementation

| ID | Claim | Acceptance |
|---|---|---|
| R3-01 | primitive volume/mass are analytic | relative error `< 1e-12` |
| R3-02 | primitive inertia is analytic | relative error `< 1e-12` |
| R3-03 | free fall follows constant acceleration | position and velocity relative error `< 1%` before contact |
| R3-04 | time-step refinement converges | error decreases for `dt`, `dt/2`, `dt/4` |
| R3-05 | quaternion remains normalized | norm error `< 1e-12` after integration |
| R3-06 | sphere collision conserves momentum | relative drift `< 1e-12` with gravity off |
| R3-07 | closed container prevents persistent penetration | final penetration `< 1e-6 m` |
| R3-08 | deterministic replay is exact | serialized state byte-identical |
| R3-09 | no invalid state | NaN and infinity count exactly zero |
| R3-10 | collision solve does not create unbounded energy | post-impact energy within restitution/friction expectation |

All tests emit measured values. Tolerances are not relaxed in response to
failure.

## Interactive acceptance

- Add, remove, select, reset, and drop bodies.
- Sphere, box, capsule, and cylinder choices.
- Edit density, scale, position, restitution, and friction with SI units.
- Display mass, pose, linear/angular velocity, force, torque, collision impulse,
  kinetic/potential energy, and contact count for the selected body.
- Render orientation and partial submergence without implying buoyancy.
- Export initial descriptions plus current body state and diagnostics.
