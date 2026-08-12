import type { SceneDescription, Vec3 } from "./model";
import { sceneDamBreakBox } from "./initial-fluid";
import { editorFluidLattice, fluidBrickIndexAt } from "./editor-fluid";

/**
 * Canonical initial-water geometry.
 *
 * Every occupied body is an axis-aligned region in container coordinates:
 * x/y/z all run from zero at the container minimum to one at its maximum.
 * A settled fill, one analytic dam, disconnected dams and a painted blob are
 * therefore the same geometry; only the compatibility codec used to write the
 * current document schema differs.
 */
export interface InitialFluidRegion {
  readonly min: Vec3;
  readonly max: Vec3;
  readonly codec: "tank-fill" | "dam-box" | "brick";
}

export interface InitialFluidLayout {
  readonly regions: readonly InitialFluidRegion[];
  readonly brickSeedsAdditive: boolean;
}

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

function baseRegion(scene: SceneDescription): InitialFluidRegion | undefined {
  if (scene.systems?.fluid === false) return undefined;
  if (scene.fluid.initialCondition === "tank-fill") {
    const fill = clamp01(scene.container.fillFraction);
    return fill > 0 ? {
      min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: fill, z: 1 }, codec: "tank-fill",
    } : undefined;
  }
  const box = sceneDamBreakBox(scene);
  return box.max.x > box.min.x && box.max.y > box.min.y && box.max.z > box.min.z
    ? { min: { ...box.min }, max: { ...box.max }, codec: "dam-box" }
    : undefined;
}

/** Capture every initial body without giving any storage form geometric priority. */
export function initialFluidLayout(scene: SceneDescription): InitialFluidLayout {
  const regions: InitialFluidRegion[] = [];
  const seeds = scene.fluid.initialBrickSeeds_m ?? [];
  const additive = seeds.length > 0 && scene.fluid.initialBrickSeedsAdditive === true;
  const base = baseRegion(scene);
  if (base && (seeds.length === 0 || additive)) regions.push(base);

  if (seeds.length > 0) {
    const lattice = editorFluidLattice(scene);
    const occupied = new Set<string>();
    for (const seed of seeds) {
      const index = fluidBrickIndexAt(lattice, seed);
      if (!index) continue;
      const key = `${index.x}:${index.y}:${index.z}`;
      if (occupied.has(key)) continue;
      occupied.add(key);
      const brickFraction = {
        x: lattice.brickSize_m.x / scene.container.width_m,
        y: lattice.brickSize_m.y / scene.container.height_m,
        z: lattice.brickSize_m.z / scene.container.depth_m,
      };
      regions.push({
        min: {
          x: index.x * brickFraction.x,
          y: index.y * brickFraction.y,
          z: index.z * brickFraction.z,
        },
        max: {
          x: Math.min(1, (index.x + 1) * brickFraction.x),
          y: Math.min(1, (index.y + 1) * brickFraction.y),
          z: Math.min(1, (index.z + 1) * brickFraction.z),
        },
        codec: "brick",
      });
    }
  }
  return { regions, brickSeedsAdditive: additive };
}

const dimensions = (scene: SceneDescription) => [
  scene.container.width_m, scene.container.height_m, scene.container.depth_m,
] as const;
const axes = ["x", "y", "z"] as const;

/**
 * Write canonical geometry into a resized document.
 *
 * This is deliberately the only container-scale codec. The geometry itself is
 * never scaled case-by-case: normalized regions already move and resize with
 * the container. Legacy fields are merely projections consumed by the current
 * solver backends.
 */
export function applyInitialFluidLayout(
  scene: SceneDescription,
  layout: InitialFluidLayout,
): SceneDescription {
  const extents = dimensions(scene);
  const base = layout.regions.find((region) => region.codec !== "brick");
  const brickRegions = layout.regions.filter((region) => region.codec === "brick");

  if (base?.codec === "tank-fill") {
    scene.fluid.initialCondition = "tank-fill";
    scene.container.fillFraction = clamp01(base.max.y - base.min.y);
    delete scene.fluid.initialDamBreakDimensions_m;
    delete scene.fluid.initialDamBreakOrigin_m;
  } else if (base?.codec === "dam-box") {
    scene.fluid.initialCondition = "dam-break";
    const size = {} as Vec3, origin = {} as Vec3;
    axes.forEach((axis, index) => {
      size[axis] = Math.max(0, base.max[axis] - base.min[axis]) * extents[index]!;
      origin[axis] = base.min[axis] * extents[index]!;
    });
    scene.fluid.initialDamBreakDimensions_m = size;
    if (axes.every((axis) => Math.abs(origin[axis]) < 1e-12)) {
      delete scene.fluid.initialDamBreakOrigin_m;
    } else scene.fluid.initialDamBreakOrigin_m = origin;
    scene.container.fillFraction = clamp01(
      (base.max.x - base.min.x) * (base.max.y - base.min.y) * (base.max.z - base.min.z));
  }

  if (brickRegions.length > 0) {
    // A region is geometry, not a promise to keep one seed. DETAIL changes the
    // brick lattice, so rasterize the same normalized volume onto every new
    // brick it overlaps. A ×2 detail step turns one old brick into 2³ new
    // bricks; ÷2 naturally deduplicates several old regions into one coarse
    // brick. This is the operation the former point-scaling implementation
    // missed, leaving twin-dam seeds separated at higher resolution.
    const lattice = editorFluidLattice(scene);
    const fraction = {
      x: lattice.brickSize_m.x / scene.container.width_m,
      y: lattice.brickSize_m.y / scene.container.height_m,
      z: lattice.brickSize_m.z / scene.container.depth_m,
    };
    const seeds = new Map<string, Vec3>();
    for (const region of brickRegions) {
      const range = (axis: keyof Vec3, count: number) => {
        const epsilon = 1e-10;
        const first = Math.max(0, Math.floor((region.min[axis] + epsilon) / fraction[axis]));
        const last = Math.min(count - 1, Math.ceil((region.max[axis] - epsilon) / fraction[axis]) - 1);
        return { first, last };
      };
      const xr = range("x", lattice.bricks.x), yr = range("y", lattice.bricks.y), zr = range("z", lattice.bricks.z);
      for (let z = zr.first; z <= zr.last; z += 1)
        for (let y = yr.first; y <= yr.last; y += 1)
          for (let x = xr.first; x <= xr.last; x += 1)
            seeds.set(`${x}:${y}:${z}`, {
              x: lattice.origin_m.x + (x + 0.5) * lattice.brickSize_m.x,
              y: lattice.origin_m.y + (y + 0.5) * lattice.brickSize_m.y,
              z: lattice.origin_m.z + (z + 0.5) * lattice.brickSize_m.z,
            });
    }
    scene.fluid.initialBrickSeeds_m = [...seeds.values()];
    if (layout.brickSeedsAdditive) scene.fluid.initialBrickSeedsAdditive = true;
    else delete scene.fluid.initialBrickSeedsAdditive;
  } else {
    delete scene.fluid.initialBrickSeeds_m;
    delete scene.fluid.initialBrickSeedsAdditive;
  }
  return scene;
}

/** Volume share of the container, with overlapping regions intentionally additive. */
export function initialFluidLayoutVolumeFraction(layout: InitialFluidLayout): number {
  return layout.regions.reduce((sum, region) => sum
    + Math.max(0, region.max.x - region.min.x)
      * Math.max(0, region.max.y - region.min.y)
      * Math.max(0, region.max.z - region.min.z), 0);
}
