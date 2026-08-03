# Preview scenes

Throwaway scene modules for looking at one thing through the production
dry-scene render path.

    OUT=/tmp/my-shot && mkdir -p $OUT && \
    WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
    FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/<name>.ts \
    FLUID_SVO_DRY_FRAME_WIDTH=900 FLUID_SVO_DRY_FRAME_HEIGHT=520 \
    FLUID_SVO_DRY_FRAME_WARMUPS=1 FLUID_SVO_DRY_FRAME_CYCLES=1 \
    FLUID_SVO_DRY_FRAME_OUT=$OUT/frame.json \
    node --import tsx tools/benchmark-svo-dry-frame-gpu.ts

The frame lands at `$OUT/reference.png` and the timings at `$OUT/frame.json`.

A module exports `createScene(): SceneDescription` and optionally
`camera: Partial<CameraState>`.

## Build on `heroPreviewScene`, never on the scene factory

```ts
import { heroPreviewScene, heroPreviewCamera } from "./hero-still";
import { myThing } from "../../lib/voxel-scenery/my-thing";

export const createScene = () => heroPreviewScene(myThing({ seed: 7 }));
export const camera = { ...heroPreviewCamera(), distance_m: 0.9 };
```

A scene factory such as `createHeroGardenHoseScene` returns a document *body*.
The art-directed environment is attached by the catalog on the way out
(`sceneDocument` in `lib/scene-definition.ts`), so a preview that calls the
factory directly renders the porcelain garden under the default set's dark teal
sky — and every judgement made from that render about colour, value or contrast
is a judgement about the wrong lighting.

`heroPreviewScene` also clones the scenery graph before appending. The graph is a
module-level constant shared by every document the factory produces; pushing
nodes onto it edits the hero scene itself for the rest of the process.
