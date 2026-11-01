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
FVX bakes, plus a manifest for the generated asset catalog. The tower has a wider, more strongly tapered hollow whitewashed masonry shell, boolean-cut
door/window openings, projecting sills and lintels, cornices, lantern mullions,
a dark watch room between two gallery levels with radial corbels and post collars, standing roof seams and
a lightning conductor. A broad rock summit, descending tilted slabs, low tidal outcrops, intersecting fractures and fine bedding
are modeled geometry. Low tidal boulders have full shoulders and rounded crowns cut by angular bedding, rather than pointed cone profiles. The dry rock mesh samples `lighthouse_terrain.py` at
6.25 cm; the runtime collision heightfield samples the same function at 25 cm.
The original fluid volumes, fluid grid and solver settings are retained. The default camera now frames the wider island from a low viewing angle.
Whitewash color, roughness and normal maps are generated deterministically in
Blender and carried through glTF/FVM. Existing photographed rock textures remain
in use. These are surface properties, with no illumination baked into them.
The runtime lantern keeps
its real open mullions and bulb; opaque FVX does not substitute solid fake glass.
The rock visual mesh shares the physical terrain function at finer resolution.
The tower's actual FVX geometry is also resampled into the fluid's 25 cm
collision field, preserving the masonry shell and hollow interior. The 24 m
domain height encloses its complete eighteen-metre silhouette.

Rebuild the static assets from an authored preview `.blend`:

```sh
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/export-atlantic-lighthouse.py
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/bake_mesh_volume.py -- --collection atlantic-lighthouse
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/bake_surface_mapping.py -- --collection atlantic-lighthouse --asset atlantic-lighthouse-rock --output-dir artifacts/atlantic-lighthouse/materials --publish
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" -b --python-exit-code 1 --python tools/blender/bake_surface_mapping.py -- --collection atlantic-lighthouse --asset atlantic-lighthouse-tower --output-dir artifacts/atlantic-lighthouse/materials --publish
npm run generate:assets
node --import tsx tools/blender/bake-atlantic-collider.ts
```

Runtime FVX pitches are 2.5 cm for the tower and 8 cm for the rock. Import placement
includes the sampled-volume bounds offset, so the visible rock, lighthouse
footing and physical shore remain aligned. No baked water mesh is imported into
the app.

The reference update uses a low warm directional sun from the visible right side with cool hemisphere fill. The sun direction is shared with Blender. The island summit supports the wider tower and attached doorway surround without a narrow pedestal; the collision baker derives its crop from the actual FVX bounds, including the entry steps.
The shared SVO environment evaluates darker dense cloud undersides and brighter
thin edges, including the same response in water reflections. It remains an
analytic distant sky, not the Cycles stage's volumetric atmosphere. Cloud-free
scenes retain their previous evaluation. `tests/environment-clouds.test.ts`
compares the actual GPU evaluator against its CPU reference, including rough
reflections, and guards against an overcast sky collapsing into a flat color.

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
procedural microsurface ripples, chipped whitewashed masonry and sculpted rock outcrops, glazing,
iron gallery railings, maritime haze, and a break in the storm lighting.
Foam is a procedural surface-aeration shader on the simulated mesh; it is not
a secondary whitewater particle simulation. Fluid resolution is 25 cm, so
sub-cell spray is not resolved. Those limitations must not be described as a
fully resolved ocean or physically simulated foam.
