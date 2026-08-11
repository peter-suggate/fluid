import type { SceneDescription } from "./model";
import { pondVesselPlanCurve, type PondVesselSpec } from "./voxel-scenery/pond-vessel";

/**
 * Three dials over a pond vessel's coping, phrased the way a rim reads.
 *
 * The rim is not scenery: it is the vessel's own crest section, baked into the
 * procedural terrain (`scene.terrain.procedural`) and consulted by every
 * generator that beds against the pond. So an edit here rewrites the vessel in
 * *both* places the document holds it — the scenery graph's vessel table and
 * the terrain description — and everything downstream re-derives: the ground,
 * the beds, the wading path. The dials:
 *
 *  - **height** — the crest above the plaster.
 *  - **width** — half-width of the coping band, log-spaced: a rim reads by its
 *    ratio to the pond, not by millimetres.
 *  - **rough** — the hand-formed axis: surface relief and the swelling and
 *    narrowing of the section as it runs, moved together. Zero is a machined
 *    extrusion; the authored hero rim sits exactly on this line, so selecting
 *    it changes nothing until a dial moves.
 *
 * The plan curve is deliberately not a dial: the stones, the path and the
 * terraces are all laid against it, and the split boulders now stand at baked
 * world positions — a rim that wandered out from under them would strand the
 * whole bank.
 */
export interface RimDials {
  /** Crest height above the plaster, in [0, 1]. */
  readonly height: number;
  /** Coping band width, in [0, 1]; log-spaced. */
  readonly width: number;
  /** Machined extrusion (0) to hand-formed (1). */
  readonly rough: number;
}

// Crest height in metres. The hero rim is 55 mm, interior at 0.35.
const HEIGHT_MIN_M = 0.02;
const HEIGHT_MAX_M = 0.12;
// Half-width in metres, log-spaced; the hero's 30 mm reads back at exactly 0.5.
const WIDTH_MIN_M = 0.012;
const WIDTH_MAX_M = 0.075;
// The hand-formed axis: relief and the two section variations move on fixed
// rays from zero, chosen so the authored hero vessel (relief 2.5 mm, section
// variation 0.16 / 0.22) sits exactly on the line at rough = 0.3125 — the
// readback is exact and the first touch of the dial never snaps the shape.
const RELIEF_MAX_M = 0.008;
const SECTION_HEIGHT_VARIATION_MAX = 0.512;
const SECTION_WIDTH_VARIATION_MAX = 0.704;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const unlerp = (a: number, b: number, value: number) => clamp01((value - a) / (b - a));

/** The vessels a scene declares, in the graph's own order. */
export function sceneVessels(scene: SceneDescription): readonly [string, PondVesselSpec][] {
  return Object.entries(scene.scenery?.vessels ?? {});
}

/** The named vessel, or undefined. */
export function sceneVessel(scene: SceneDescription, name: string): PondVesselSpec | undefined {
  return scene.scenery?.vessels?.[name];
}

/** The dial positions a vessel's stored rim reads back as. */
export function rimDials(spec: PondVesselSpec): RimDials {
  return {
    height: unlerp(HEIGHT_MIN_M, HEIGHT_MAX_M, spec.rimHeight_m),
    width: clamp01(Math.log(spec.rimHalfWidth_m / WIDTH_MIN_M) / Math.log(WIDTH_MAX_M / WIDTH_MIN_M)),
    rough: clamp01(spec.relief_m / RELIEF_MAX_M),
  };
}

function specForDials(current: PondVesselSpec, dials: RimDials): PondVesselSpec {
  const rough = clamp01(dials.rough);
  return {
    ...current,
    rimHeight_m: lerp(HEIGHT_MIN_M, HEIGHT_MAX_M, clamp01(dials.height)),
    rimHalfWidth_m: WIDTH_MIN_M * (WIDTH_MAX_M / WIDTH_MIN_M) ** clamp01(dials.width),
    relief_m: RELIEF_MAX_M * rough,
    sectionHeightVariation: SECTION_HEIGHT_VARIATION_MAX * rough,
    sectionWidthVariation: SECTION_WIDTH_VARIATION_MAX * rough,
  };
}

/**
 * The scene with the named vessel's rim set to `dials`; a no-op elsewhere.
 *
 * The vessel is one authority held in two places: the scenery graph's table
 * (what the generators bed against) and the terrain description (what the
 * ground *is*). When the terrain's procedural spec is this same vessel — the
 * only arrangement the presets author — both are rewritten together, so the
 * beds can never ring a crest the ground no longer carries.
 */
export function withRimDials(
  scene: SceneDescription,
  name: string,
  dials: RimDials,
): SceneDescription {
  const current = sceneVessel(scene, name);
  if (!current || !scene.scenery) return scene;
  const next = specForDials(current, dials);
  const scenery = {
    ...scene.scenery,
    vessels: { ...scene.scenery.vessels, [name]: next },
  };
  const procedural = scene.terrain?.procedural;
  const terrain = scene.terrain && procedural?.kind === "pond-vessel"
    && JSON.stringify(procedural.spec) === JSON.stringify(current)
    ? { ...scene.terrain, procedural: { ...procedural, spec: next } }
    : scene.terrain;
  return { ...scene, scenery, terrain };
}

/** The rim band's outer reach from the crest centreline, with the grab margin. */
export function rimBandReach_m(spec: PondVesselSpec): number {
  return spec.rimHalfWidth_m * (1 + spec.sectionWidthVariation) + 0.006;
}

/**
 * The vessel whose coping band contains the plan point, or undefined.
 *
 * Distance is measured to the vessel's own plan curve — the wandered crest
 * centreline every consumer lays against — so what is clickable is the band
 * that is actually drawn, not an ellipse the wobble left behind.
 */
export function vesselRimAt(
  scene: SceneDescription,
  x: number,
  z: number,
): string | undefined {
  for (const [name, spec] of sceneVessels(scene)) {
    if (planDistanceTo(pondVesselPlanCurve(spec), x, z) <= rimBandReach_m(spec)) return name;
  }
  return undefined;
}

/** Unsigned plan distance from a point to a closed polyline. */
function planDistanceTo(
  curve: readonly (readonly [number, number])[],
  x: number,
  z: number,
): number {
  let nearest = Infinity;
  for (let index = 0; index < curve.length; index += 1) {
    const [ax, az] = curve[index]!;
    const [bx, bz] = curve[(index + 1) % curve.length]!;
    const dx = bx - ax, dz = bz - az;
    const lengthSquared = dx * dx + dz * dz;
    const t = lengthSquared < 1e-12 ? 0
      : Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / lengthSquared));
    nearest = Math.min(nearest, Math.hypot(x - (ax + t * dx), z - (az + t * dz)));
  }
  return nearest;
}

/**
 * The rim state of every vessel in the scene, as one compact string:
 * `name~height,width,rough` per vessel, ";"-joined. Same contract as the
 * canopy and stone queries: three 3-decimal numbers reproduce anything the
 * flyout can reach, and a carried value re-reads as the same string.
 */
export function sceneRimQuery(scene: SceneDescription): string {
  const round = (value: number) => String(Math.round(value * 1000) / 1000);
  return sceneVessels(scene)
    .map(([name, spec]) => {
      const dials = rimDials(spec);
      return `${name}~${round(dials.height)},${round(dials.width)},${round(dials.rough)}`;
    })
    .join(";");
}

/** Apply a `sceneRimQuery` string; unknown vessels and malformed runs are skipped. */
export function withSceneRimQuery(scene: SceneDescription, query: string): SceneDescription {
  let next = scene;
  for (const run of query.split(";")) {
    const separator = run.lastIndexOf("~");
    if (separator <= 0) continue;
    const name = run.slice(0, separator);
    const values = run.slice(separator + 1).split(",").map(Number);
    if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) continue;
    if (sceneVessel(next, name) === undefined) continue;
    next = withRimDials(next, name, {
      height: values[0]!,
      width: values[1]!,
      rough: values[2]!,
    });
  }
  return next;
}
