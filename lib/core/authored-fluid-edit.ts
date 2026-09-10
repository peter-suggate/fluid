import type { InitialLiquidBox, InitialLiquidVolume, SceneDescription } from "./model";
import { initialFluidLayout } from "./initial-fluid-layout";

/** Geometry edited in place by Sparse CM12; unrelated seed inputs keep their reset semantics. */
export function authoredFluidGeometryKey(scene: SceneDescription): string {
  const f = scene.fluid;
  return JSON.stringify([scene.container.fillFraction, f.initialCondition,
    f.initialDamBreakDimensions_m, f.initialDamBreakOrigin_m, f.initialBrickSeeds_m,
    f.initialBrickSeedsAdditive, f.initialLiquidVolumes]);
}

/** Resolve every editor storage form to world-space primitives, without rasterizing cells. */
export function authoredFluidVolumes(scene: SceneDescription): InitialLiquidVolume[] {
  const c = scene.container;
  const boxes = initialFluidLayout(scene).regions.map(({ min, max }): InitialLiquidBox => ({
    shape: "box",
    min_m: { x: (min.x - 0.5) * c.width_m, y: min.y * c.height_m, z: (min.z - 0.5) * c.depth_m },
    max_m: { x: (max.x - 0.5) * c.width_m, y: max.y * c.height_m, z: (max.z - 0.5) * c.depth_m },
  }));
  // A painted rectangular body may contain thousands of bricks. Merge touching
  // boxes before dispatch so a move costs one shape operation per solid region,
  // rather than one complete topology transaction per painted brick.
  let merged = boxes;
  for (const axis of ["x", "y", "z"] as const) {
    const others = (["x", "y", "z"] as const).filter(a => a !== axis);
    const groups = new Map<string, InitialLiquidBox[]>();
    for (const box of merged) {
      const key = others.map(a => `${box.min_m[a]}:${box.max_m[a]}`).join("|");
      const group = groups.get(key) ?? []; group.push(box); groups.set(key, group);
    }
    merged = [];
    for (const group of groups.values()) {
      group.sort((a, b) => a.min_m[axis] - b.min_m[axis]);
      let previous: InitialLiquidBox | undefined;
      for (const box of group) {
        if (previous && Math.abs(previous.max_m[axis] - box.min_m[axis]) < 1e-10) {
          previous.max_m[axis] = box.max_m[axis];
        } else { previous = box; merged.push(box); }
      }
    }
  }
  return [...merged, ...(scene.fluid.initialLiquidVolumes ?? [])];
}

/** Replace edited source footprints in the current field, retaining the timeline.
 * Water that has flowed away from an authored footprint remains simulated water.
 */
export function authoredFluidEdits(before: SceneDescription, after: SceneDescription): {
  operation: "add" | "remove"; volume: InitialLiquidVolume;
}[] {
  if (authoredFluidGeometryKey(before) === authoredFluidGeometryKey(after)) return [];
  const oldVolumes = new Map(authoredFluidVolumes(before).map(v => [JSON.stringify(v), v]));
  const newVolumes = new Map(authoredFluidVolumes(after).map(v => [JSON.stringify(v), v]));
  return [
    ...[...oldVolumes].filter(([key]) => !newVolumes.has(key)).map(([, volume]) => ({ operation: "remove" as const, volume })),
    ...[...newVolumes].filter(([key]) => !oldVolumes.has(key)).map(([, volume]) => ({ operation: "add" as const, volume })),
  ];
}

/** Eight existing injection words also represent boxes, cylinders, rings and cut spheres. */
export function packAuthoredFluidVolume(volume: InitialLiquidVolume, operation: "add" | "remove",
  cellSize: number, origin: readonly number[]) {
  const xyz = (v: { x: number; y: number; z: number }): [number, number, number] => [v.x, v.y, v.z];
  let center: number[], radius: [number, number, number], extra = 0, mode: number;
  if (volume.shape === "box") {
    const lo = xyz(volume.min_m), hi = xyz(volume.max_m);
    center = lo.map((v, a) => (v + hi[a]!) / 2);
    radius = lo.map((v, a) => (hi[a]! - v) / (2 * cellSize)) as typeof radius;
    mode = operation === "add" ? 3 : 6;
  } else {
    center = xyz(volume.center_m);
    const r = volume.radius_m / cellSize;
    radius = [r, r, r];
    mode = operation === "add" ? 1 : 5;
    if (volume.shape === "torus") {
      extra = volume.tubeRadius_m / cellSize;
      radius = [r + extra, extra, r + extra];
      mode = operation === "add" ? 4 : 7;
    } else if (volume.shape === "cylinder") {
      radius[2] = volume.halfHeight_m / cellSize;
      mode = operation === "add" ? 8 : 9;
    } else if (volume.shape === "hemisphere") {
      const normal = xyz(volume.outwardNormal), length = Math.hypot(...normal);
      radius = length > 1e-12 ? normal.map(v => v / length) as typeof radius : [1, 0, 0];
      extra = r;
      mode = operation === "add" ? 10 : 11;
    }
  }
  if (![...center, ...radius, extra, cellSize, ...origin].every(Number.isFinite)
    || !(cellSize > 0) || (volume.shape === "hemisphere" ? !(extra > 0) : radius.some(r => !(r > 0)))
    || (volume.shape === "torus" && !(volume.tubeRadius_m > 0 && volume.tubeRadius_m < volume.radius_m))) {
    throw new RangeError("Authored fluid edits require finite coordinates and positive shape dimensions.");
  }
  return { center: center.map((v, a) => (v - origin[a]!) / cellSize) as [number, number, number],
    radius, extra, mode };
}
