# Atlantic lighthouse — live scene and Blender source assets

The live preset is `atlantic-lighthouse`, created by
`lib/core/atlantic-lighthouse-scene.ts`. Its registered fluid method is narrow-band
FLIP. A rounded, eighteen-metre-wide elevated bore starts at rest and
collapses under gravity onto a real collision heightfield. Its upper surface
starts at 6.5 m, above a 1.6 m sea level. The finite 48 × 24 × 48 m physical
domain places lateral walls outside the photographed region, well beyond the
island. The background sea fills the domain at rest, avoiding an artificial
fast-moving dry front around a finite pool. This is a 2.5-second coastal impact
shot, not an infinite ocean.
The existing solver does not support open walls together with sculpted terrain
and tower collisions, so this scene uses its standard bounded domain.
The scene contains an initial sea and crest; it has no prescribed
animated water surface. The app draws the Blender-authored lighthouse and rock
island as native sampled-volume field programs while NB-FLIP water runs live.

Static geometry lives in `public/assets/atlantic-lighthouse`: two GLBs and their
FVX bakes, plus a manifest for the generated asset catalog. The tower has
individually modelled granite masonry, cornices, door, windows, lantern mullions,
gallery railings and a lightning conductor. A recessed lime-mortar shell closes
the stone joints while preserving the tower's hollow cavity; projecting window
sills and lintels break up the otherwise uniform wall. Stone/mineral face palettes survive
the GLB/FVX conversion. A portable tangent normal map carries the basalt grain
and irregular fracture shader into the native shading-normal channel, separately
from the geometric distance gradient used by physics. The runtime lantern keeps
its real open mullions and bulb; opaque FVX does not substitute solid fake glass.
The rock visual mesh shares the exact physical terrain sampling function.
The tower's actual FVX geometry is also resampled into the fluid's 25 cm
collision field, preserving the masonry shell and hollow interior. The 24 m
domain height encloses its complete eighteen-metre silhouette.

Rebuild the static assets from an authored preview `.blend`:

```sh
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/export-atlantic-lighthouse.py
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/bake-atlantic-normal.py
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/bake_mesh_volume.py -- --collection atlantic-lighthouse
npm run generate:assets
node --import tsx tools/blender/bake-atlantic-collider.ts
```

Runtime pitches are 4 cm for the tower and 12.5 cm for the rock. Import placement
includes the sampled-volume bounds offset, so the visible rock, lighthouse
footing and physical shore remain aligned. No baked water mesh is imported into
the app.

Export a simulated frame using the shared `export-nbflip-water.ts` pipeline,
then render the OBJ in Blender:

```sh
node --import tsx tools/blender/export-nbflip-water.ts --scene=atlantic-lighthouse --time=1
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/atlantic_lighthouse.py -- --water-mesh artifacts/water-scenes/atlantic-lighthouse/water.obj --output artifacts/water-scenes/atlantic-lighthouse/render.png --width 1600 --height 1100 --samples 128
```

The renderer requires a water mesh and fails if one is missing. Application
coordinates `(x,y,z)` map to Blender `(x,-z,y)`, in metres. It imports the OBJ
without an additional axis conversion, welds duplicated triangle vertices and
recomputes normals. The rock shelf repeats the exact collision heightfield.
The .blend remains editable and the adjacent JSON records input provenance.

Rendering uses Cycles with dielectric water at IOR 1.333, volume absorption,
procedural microsurface ripples, weathered individual masonry blocks, glazing,
iron gallery railings, maritime haze, and a break in the storm lighting.
Foam is a procedural surface-aeration shader on the simulated mesh; it is not
a secondary whitewater particle simulation. Fluid resolution is 25 cm, so
sub-cell spray is not resolved. Those limitations must not be described as a
fully resolved ocean or physically simulated foam.
