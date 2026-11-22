# Broadside galleon

The Blender source adapts tevy’s 44 m Sailing Galleon v1 (LOD 0), released under CC0. The lower courses are bundled at the yards. Two shortened topsails and a split black pennant leave the gun decks and rigging exposed. The Blender file contains the packed materials. [Original model](https://tevy.dev/models/galleon) · [CC0](https://creativecommons.org/publicdomain/zero/1.0/)

Export the ship from Blender as `public/assets/galleons/sailing-galleon.glb` with materials and Y up. Run `python tools/galleons/bake.py` with numpy, scipy, Pillow and libigl installed, then `npm run generate:assets` and `node --import tsx tools/galleons/collider.ts`. Both instances share the source volume and texture charts. The ship bake samples the mesh at 10 cm and gives open surfaces 7 cm of thickness on either side. The starboard ratlines remain individually pickable at that spacing. Shading interpolates Blender’s vertex normals at the nearest point on each triangle and uses a 2048-pixel material atlas. The ocean uses its own 1 m simulation grid.

## Render the preview

Capture the water and shot positions after 44 simulation steps with `tools/gpu/fluid-gpu-run.sh check node --import tsx tools/galleons/capture.ts /tmp/galleon-frame`. The capture reads the production Metal solver’s vertex level set and uses the live launcher simulation.

Open `assets/galleons/broadside.blend` in Blender’s background mode and run `--python tools/galleons/render.py -- /tmp/galleon-frame /path/to/sky.hdr public/assets/galleons/preview.png`. The still uses the captured ocean surface and adds fine water normals and volumetric powder smoke in Cycles. The sky is Poly Haven’s CC0 Kloofendal 48d Partly Cloudy Pure Sky. [Sky source](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
