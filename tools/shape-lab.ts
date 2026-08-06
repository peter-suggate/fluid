/**
 * Draw a procedural form straight to a PNG, with no GPU and no voxelization.
 *
 *     node --import tsx tools/shape-lab.ts wall   out/wall.png
 *     node --import tsx tools/shape-lab.ts canopy out/canopy.png
 *
 * ## Why this exists
 *
 * The hero garden's forms were being judged by building a scene, publishing a
 * few hundred primitives, taking the exclusive GPU lock, voxelizing at 6.25 mm
 * and rendering a frame — minutes per look, on a picture where the voxel
 * terracing is loud enough to hide the thing being judged. Two art-direction
 * passes were spent tuning parameters inside functions that could not express
 * the target shape *at all*, and neither pass noticed, because the loop was too
 * slow to ask the only question that mattered: what curve is this function
 * actually drawing?
 *
 * So this draws the curve. A profile is a function of one variable and it can be
 * plotted; a field program is an SDF and it can be sphere-traced on the CPU at a
 * few hundred pixels in under a second. Neither needs the renderer, and both can
 * be put beside the reference plate while the form is still an argument about
 * numbers.
 *
 * The two modes are deliberately the two shapes the hero scene gets wrong:
 *
 *  - **wall** plots ground height against signed plan distance, so "flat top,
 *    steep face, crest you can see" is a property of a drawing rather than of a
 *    paragraph. It plots the smoothstep the vessel ships beside it, because the
 *    whole point is that a smoothstep has no flat and no crest and never could.
 *  - **canopy** sphere-traces an `SvoFieldProgram` tape. A nested `scatter` is
 *    self-similar by construction and the only honest way to know how many
 *    levels read at a given size is to look at it.
 *
 * This is a *lab*, not a gate. It draws the analytic form; the renderer draws
 * what the octree made of it, and the two differ by exactly the resampling the
 * lab exists to see past.
 */
import { encodeRgbPng } from "../lib/png-codec";
import type { Vec3 } from "../lib/model";
import {
  evaluateSvoFieldProgram,
  svoFieldProgramExtent_m,
  svoFieldProgramFeatureRadius_m,
  validateSvoFieldProgram,
  type SvoFieldProgram,
} from "../lib/svo-field-program";
import { writeFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------

interface Canvas {
  readonly width: number;
  readonly height: number;
  readonly rgb: Uint8Array;
}

function canvas(width: number, height: number, fill: readonly [number, number, number]): Canvas {
  const rgb = new Uint8Array(width * height * 3);
  for (let index = 0; index < width * height; index += 1) {
    rgb[3 * index] = fill[0];
    rgb[3 * index + 1] = fill[1];
    rgb[3 * index + 2] = fill[2];
  }
  return { width, height, rgb };
}

function plot(target: Canvas, x: number, y: number, colour: readonly [number, number, number]): void {
  const px = Math.round(x), py = Math.round(y);
  if (px < 0 || py < 0 || px >= target.width || py >= target.height) return;
  const index = 3 * (py * target.width + px);
  target.rgb[index] = colour[0];
  target.rgb[index + 1] = colour[1];
  target.rgb[index + 2] = colour[2];
}

/** A line, thick enough to read against a filled region. */
function stroke(
  target: Canvas,
  from: readonly [number, number],
  to: readonly [number, number],
  colour: readonly [number, number, number],
  weight = 1,
): void {
  const steps = Math.max(1, Math.ceil(Math.hypot(to[0] - from[0], to[1] - from[1])));
  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const x = from[0] + (to[0] - from[0]) * t;
    const y = from[1] + (to[1] - from[1]) * t;
    for (let dy = -weight + 1; dy < weight; dy += 1) {
      for (let dx = -weight + 1; dx < weight; dx += 1) plot(target, x + dx, y + dy, colour);
    }
  }
}

function save(target: Canvas, path: string): void {
  writeFileSync(path, encodeRgbPng({ width: target.width, height: target.height, rgb: target.rgb }));
  console.log(`wrote ${path} (${target.width}x${target.height})`);
}

// ---------------------------------------------------------------------------
// Mode: wall — a height profile against signed plan distance
// ---------------------------------------------------------------------------

/** The smoothstep the vessel ships. Plotted only so the comparison is on one sheet. */
const smoothstep = (t: number): number => {
  const clamped = Math.min(1, Math.max(0, t));
  return clamped * clamped * (3 - 2 * clamped);
};

export interface WallProfile {
  /** Rise from the ground outside to the plateau on top. */
  readonly rise_m: number;
  /** Radius of the roll where the plateau turns over into the face. */
  readonly crestRadius_m: number;
  /** Radius of the fillet where the face meets the ground. */
  readonly footRadius_m: number;
  /**
   * How far the face leans out from vertical, in radians. Zero is a plumb wall;
   * a real formed wall carries a few degrees so it can be struck.
   */
  readonly batter_rad: number;
}

/**
 * Height above the outside ground at signed plan distance `d`, where `d = 0` is
 * the **crest line** — the edge of the plateau — and `d` runs positive outward.
 *
 * Same convention as the pond's coping, whose rail is its crest centreline, so a
 * bed outline and a coping rail mean the same thing and can be offset from each
 * other without a conversion.
 *
 * Piecewise, and that is the entire point. Reading outward from the plateau: a
 * circular roll of `crestRadius_m`, a straight face leaning `batter_rad` off
 * vertical, a circular fillet of `footRadius_m`, then the ground. Four authored
 * lengths, each controlling one visible feature and nothing else — where a
 * smoothstep has a single band width that moves the crest, the face and the foot
 * together and can produce none of them.
 *
 * Both arcs are tangent to the face and to their own flat by construction, so
 * the surface is C¹ everywhere and what the eye reads as an edge is the
 * *curvature* jump at each tangent point. That is what a formed edge is: a
 * casting has no zero-radius arris either.
 *
 * `batter_rad` must be positive, and that is a property of the medium rather
 * than a style choice — a plumb face is multivalued in `d` and a heightfield
 * cannot carry it. Real formed walls are battered anyway, or they could not be
 * struck.
 */
export function wallProfile_m(profile: WallProfile, d: number): number {
  const { rise_m } = profile;
  const batter = Math.max(0.02, profile.batter_rad);
  const cos = Math.cos(batter), sin = Math.sin(batter), tan = Math.tan(batter);
  // The two arcs eat into the rise by `r(1 - sin)` each; whatever is left is the
  // straight face. Radii too fat for the rise are shared down rather than
  // rejected, so a short wall degrades to two tangent arcs and no flat face.
  const appetite = (profile.crestRadius_m + profile.footRadius_m) * (1 - sin);
  const fit = appetite > rise_m ? rise_m / appetite : 1;
  const crest = profile.crestRadius_m * fit;
  const foot = profile.footRadius_m * fit;
  // Tangent points, walking outward from the plateau edge at the origin.
  const crestCentre = { d: 0, y: rise_m - crest };
  const crestTangent = { d: crest * cos, y: crestCentre.y + crest * sin };
  const footTangentY = foot * (1 - sin);
  const faceRun = (crestTangent.y - footTangentY) * tan;
  const footTangent = { d: crestTangent.d + faceRun, y: footTangentY };
  const footCentre = { d: footTangent.d + foot * cos, y: foot };

  if (d <= 0) return rise_m;
  if (d >= footCentre.d) return 0;
  if (d <= crestTangent.d) {
    const dx = d - crestCentre.d;
    return crestCentre.y + Math.sqrt(Math.max(0, crest * crest - dx * dx));
  }
  if (d >= footTangent.d) {
    const dx = d - footCentre.d;
    return foot - Math.sqrt(Math.max(0, foot * foot - dx * dx));
  }
  return crestTangent.y - (d - crestTangent.d) / tan;
}

function drawWall(path: string): void {
  const width = 1000, height = 460;
  const image = canvas(width, height, [250, 249, 246]);
  // World window, in metres of plan distance and metres of height.
  const from_m = -0.18, to_m = 0.10, top_m = 0.11;
  const sx = (d: number) => (d - from_m) / (to_m - from_m) * width;
  const sy = (y: number) => height - 40 - (y / top_m) * (height - 80);

  const profile: WallProfile = {
    rise_m: 0.075,
    crestRadius_m: 0.004,
    footRadius_m: 0.003,
    batter_rad: 0.09,
  };

  // Ground line and a 25 mm grid, so the drawing can be read in millimetres.
  for (let d = Math.ceil(from_m / 0.025) * 0.025; d <= to_m; d += 0.025) {
    stroke(image, [sx(d), sy(0) + 14], [sx(d), sy(top_m)], [232, 230, 224]);
  }
  for (let y = 0; y <= top_m; y += 0.025) stroke(image, [0, sy(y)], [width, sy(y)], [232, 230, 224]);
  stroke(image, [0, sy(0)], [width, sy(0)], [190, 186, 178]);

  // The vessel's current terrace: one smoothstep over its whole band.
  const band = 0.036;
  for (let px = 0; px < width; px += 1) {
    const d = from_m + (to_m - from_m) * (px / width);
    const y = profile.rise_m * smoothstep(1 - (d + band) / band);
    plot(image, px, sy(y), [206, 122, 108]);
    plot(image, px, sy(y) + 1, [206, 122, 108]);
  }

  // The piecewise wall, filled so the silhouette is what is being judged.
  for (let px = 0; px < width; px += 1) {
    const d = from_m + (to_m - from_m) * (px / width);
    const y = wallProfile_m(profile, d);
    for (let py = Math.round(sy(y)); py <= sy(0); py += 1) plot(image, px, py, [225, 223, 216]);
    plot(image, px, sy(y), [40, 44, 48]);
    plot(image, px, sy(y) + 1, [40, 44, 48]);
  }
  save(image, path);

  // And the numbers the drawing is making a claim about.
  // The honest statistic, and the reason it is spelled out.
  //
  // This used to report the slope across the *middle 70 %* of the rise, which
  // excludes the crest roll and the foot fillet — precisely the parts that make
  // a profile tubular. It read 82 degrees on a wall whose transition averaged
  // 68 and spent a quarter of its height turning over, and the frame showed a
  // tube. A metric that cannot see the defect it was built to catch is worse
  // than none, because it is believed.
  const batter = Math.max(0.02, profile.batter_rad);
  const sin = Math.sin(batter), cos = Math.cos(batter);
  const appetite = (profile.crestRadius_m + profile.footRadius_m) * (1 - sin);
  const fit = appetite > profile.rise_m ? profile.rise_m / appetite : 1;
  const arcRise = (profile.crestRadius_m + profile.footRadius_m) * fit * (1 - sin);
  const arcRun = (profile.crestRadius_m + profile.footRadius_m) * fit * cos;
  let run = 0;
  while (wallProfile_m(profile, run) > 0 && run < 1) run += 0.0001;
  let flatRun = 0;
  for (let d = from_m; d <= 0; d += 0.0005) if (wallProfile_m(profile, d) > profile.rise_m - 0.0005) flatRun += 0.0005;
  console.log(`  plateau: ${(flatRun * 1e3).toFixed(0)} mm of run within 0.5 mm of full rise`);
  console.log(`  transition: ${(run * 1e3).toFixed(1)} mm wide, of which ${(arcRun * 1e3).toFixed(1)} mm is arc`
    + ` (${(100 * arcRun / run).toFixed(0)} %)`);
  console.log(`  arcs take ${(arcRise * 1e3).toFixed(1)} mm of the ${(profile.rise_m * 1e3).toFixed(0)} mm rise`
    + ` = ${(100 * arcRise / profile.rise_m).toFixed(0)} % — under about 12 % reads as a wall, over 20 % as a tube`);
  console.log(`  average slope across the whole transition: ${(Math.atan2(profile.rise_m, run) * 180 / Math.PI).toFixed(0)} deg`);
}

// ---------------------------------------------------------------------------
// Mode: canopy — sphere-trace a field program
// ---------------------------------------------------------------------------

const KEY: Vec3 = { x: 0.48, y: 0.78, z: 0.40 };

/** A tape and where it stands, which is what a scene publishes. */
interface PlacedProgram {
  readonly program: SvoFieldProgram;
  readonly at: Vec3;
}

/**
 * Trace a *set* of placed tapes, not one.
 *
 * The canopy stopped being a single mass — see `bonsai-canopy-pads.ts` — and a
 * lab that can only draw one record cannot see the thing that changed. The
 * union is a plain `min` over the records, which is what the octree does when it
 * brackets several primitives into a brick, and the Lipschitz bound is taken
 * from whichever record is nearest so the step stays sound.
 */
function traceProgram(placed: readonly PlacedProgram[], path: string, options: {
  readonly width: number;
  readonly height: number;
  /** Half-height of the view box in metres, at the target. */
  readonly reach_m: number;
  readonly target: Vec3;
  readonly eye: Vec3;
}): void {
  for (const { program } of placed) validateSvoFieldProgram(program);
  const extent = svoFieldProgramExtent_m(placed[0].program);
  console.log(`  ${placed.length} records, ${placed[0].program.ops.length} ops each,`
    + ` extent ${extent.map((v) => (v * 1e3).toFixed(0)).join(" x ")} mm,`
    + ` finest feature ${(svoFieldProgramFeatureRadius_m(placed[0].program) * 1e3).toFixed(1)} mm`);
  const image = canvas(options.width, options.height, [206, 210, 214]);
  const forward = normalize(sub(options.target, options.eye));
  const right = normalize(cross(forward, { x: 0, y: 1, z: 0 }));
  const up = cross(right, forward);
  const distance = length(sub(options.target, options.eye));
  const field = (p: Vec3) => {
    let best = { distance_m: Infinity, lipschitz: 1 };
    for (const { program, at } of placed) {
      const value = evaluateSvoFieldProgram(program, { x: p.x - at.x, y: p.y - at.y, z: p.z - at.z });
      if (value.distance_m < best.distance_m) best = value;
    }
    return best;
  };
  for (let py = 0; py < options.height; py += 1) {
    for (let px = 0; px < options.width; px += 1) {
      const u = (px + 0.5) / options.width * 2 - 1;
      const v = 1 - (py + 0.5) / options.height * 2;
      const aspect = options.width / options.height;
      const dir = normalize(add(forward, add(
        scale(right, u * aspect * options.reach_m / distance),
        scale(up, v * options.reach_m / distance),
      )));
      let travel = 0;
      let hit = false;
      for (let step = 0; step < 160; step += 1) {
        const p = add(options.eye, scale(dir, travel));
        const value = field(p);
        const clearance = value.distance_m / Math.max(1, value.lipschitz);
        if (clearance < 2e-4) { hit = true; break; }
        travel += Math.max(2e-4, clearance);
        if (travel > distance * 3) break;
      }
      if (!hit) continue;
      const p = add(options.eye, scale(dir, travel));
      const h = 3e-4;
      const normal = normalize({
        x: field({ ...p, x: p.x + h }).distance_m - field({ ...p, x: p.x - h }).distance_m,
        y: field({ ...p, y: p.y + h }).distance_m - field({ ...p, y: p.y - h }).distance_m,
        z: field({ ...p, z: p.z + h }).distance_m - field({ ...p, z: p.z - h }).distance_m,
      });
      // Flat porcelain: a warm key, a cool sky fill, no shadow. Enough to read
      // form, and deliberately not a match for the renderer's grade.
      const lambert = Math.max(0, dot(normal, normalize(KEY)));
      const sky = 0.5 + 0.5 * normal.y;
      const value = Math.min(1, 0.30 * sky + 0.78 * lambert);
      const shade = (channel: number) => Math.round(255 * Math.min(1, value * channel) ** (1 / 2.2));
      plot(image, px, py, [shade(1.0), shade(0.985), shade(0.955)]);
    }
  }
  save(image, path);
}

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const scale = (a: Vec3, k: number): Vec3 => ({ x: a.x * k, y: a.y * k, z: a.z * k });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
const normalize = (a: Vec3): Vec3 => scale(a, 1 / Math.max(1e-12, length(a)));
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});

// ---------------------------------------------------------------------------

const [mode, out] = process.argv.slice(2);
if (mode === "wall") {
  drawWall(out ?? "wall.png");
} else if (mode === "canopy") {
  const { heroCanopyPads } = await import("./shape-lab-canopy");
  // Framed a little wider than the crown and from slightly above the plate's own
  // eye line, because the two things being judged are the silhouette between the
  // pads and the grain on their tops.
  traceProgram(heroCanopyPads(), out ?? "canopy.png", {
    width: 640, height: 400, reach_m: 0.30,
    target: { x: 0, y: 0, z: 0 },
    eye: { x: 0.34, y: 0.34, z: 0.72 },
  });
} else {
  console.error("usage: shape-lab.ts <wall|canopy> <out.png>");
  process.exit(2);
}
