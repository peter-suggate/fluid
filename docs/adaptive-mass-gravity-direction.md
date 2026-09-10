# Adaptive-mass gravity direction

The scene gravity row offers Down (−Y), Up (+Y), −X, +X, −Z and +Z alongside
its on/off switch when Adaptive Mass is selected. These are world axes, so
orbiting the camera does not change the force. Selecting a direction preserves
the vector's magnitude. While gravity is off, it updates the remembered vector
without enabling gravity. Existing diagonal vectors display as Custom direction
until the user chooses a preset. Scene history, serialization and URL state
carry both the active and remembered vectors through the existing gravity fields.

The running solver adopts the vector without resetting the liquid. The renderer
uniform key now includes X and Z as well as Y; previously a horizontal-only
edit could be missed. A vector change also invalidates cached pressure rows
for the next step, including hydrostatic theta and wall-release membership,
even when liquid density and topology remain unchanged. The resident force pass already applied acceleration per
face axis, and rigid coupling already accepts the complete vector.

## Direction assumptions found

The mixed-resolution hydrostatic pressure boundary has a specialized correction
based on a floor-connected Y-column waterline. Its previous threshold accepted
strongly tilted gravity as well as downward gravity. The correction is now
restricted to downward alignment. Other orientations retain the general
ghost-fluid boundary calculation; this change does not introduce a rotated
version of the specialized hydrostatic correction.

Presentation still includes Y-column height reconstruction and floor-film
reconstruction. These are geometric representations guarded by monotonicity
and column-validity receipts, rather than force directions. They have not been
rewritten into a gravity-aligned height representation. The directional test
below verifies free flight and live reversals, not long-term equilibrium against
every wall or rotational equivalence of rendered surfaces.

## Verification

```sh
npm run test:dawn:gravity-direction
npm run test:dawn:sparse-cm12
```

Run sequentially, with no browser GPU session. The directional test uses the
production adaptive-mass solver with mixed cell sizes and default surface
conditioning. A suspended cube accelerates along each signed axis, follows the
CM12 discrete ballistic trajectory, and responds to a live gravity reversal.
It checks finite fields, mass retention, analytic acceleration, displacement,
and transverse motion. CPU tests cover off-state direction memory and the
renderer live-update key, including no solver reset for X/Z changes.
