#!/usr/bin/env node
/**
 * Solve the hero garden's camera against the reference plate — the first
 * bullet of H0 in `docs/hero-fidelity-1000x-handoff.md`.
 *
 *   npm run solve:hero-camera
 *
 * Until this lands, every region score the fidelity gate reports is
 * contaminated: a rectangle registered on the plate's canopy addresses empty
 * backdrop in a frame shot from somewhere else, and the resulting number moves
 * when the framing moves rather than when the image improves. Solving the
 * camera is not set dressing for the gate, it is the gate's precondition.
 *
 * ---------------------------------------------------------------------------
 * Why coordinate descent on renders, rather than a PnP solve
 * ---------------------------------------------------------------------------
 * The textbook answer is to pick corresponding points and solve the pose in
 * closed form. That needs correspondences, and there are none to be had here:
 * the plate is a photographic set and the frame is a set of smooth white
 * ellipsoids that does not contain the plate's features at all. There is no
 * "same corner" to click on.
 *
 * What the two images *do* share is coarse silhouette — the pond's rim, the
 * stone mass, the canopy's blob. `edgeAlignment` scores exactly that, and it is
 * smooth enough in the camera parameters to walk. So this is a search, and the
 * only thing that makes a search affordable is that
 * `FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP` renders many framings per world build.
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   FLUID_HERO_SOLVE_WIDTH / _HEIGHT   solve resolution (default 418 x 235)
 *   FLUID_HERO_SOLVE_PASSES            descent passes (default 2)
 *   FLUID_HERO_SOLVE_OUT               working directory (default artifacts/hero-camera-solve)
 *   FLUID_HERO_SOLVE_PLATE             plate path
 *   FLUID_HERO_SOLVE_START             JSON partial CameraState to start from
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { heroGardenCamera } from "../lib/core/hero-garden-scene";
import { defaultCamera, type CameraState } from "../lib/core/model";
import { decodeRgbPng } from "../lib/core/png-codec";
import { decodeLinearPng, edgeAlignment, linearFromSrgbImage, registerToScoringGrid } from "../lib/core/hero-fidelity-score";

const environment = process.env;
const width = Number(environment.FLUID_HERO_SOLVE_WIDTH ?? 418);
const height = Number(environment.FLUID_HERO_SOLVE_HEIGHT ?? 235);
const passes = Number(environment.FLUID_HERO_SOLVE_PASSES ?? 2);
const outDirectory = environment.FLUID_HERO_SOLVE_OUT ?? "artifacts/hero-camera-solve";
const platePath = environment.FLUID_HERO_SOLVE_PLATE ?? "output/imagegen/garden-pond-hose-fill-simplified.png";

mkdirSync(outDirectory, { recursive: true });
const plate = decodeLinearPng(readFileSync(platePath));

type Axis = "azimuth_rad" | "elevation_rad" | "distance_m" | "targetX" | "targetY" | "targetZ";

interface AxisSpec {
  readonly axis: Axis;
  /** Full width of the first pass's probe, in the axis's own units. */
  readonly span: number;
  readonly minimum?: number;
  readonly maximum?: number;
}

/**
 * The order matters. Distance and azimuth move the image most, and fixing them
 * first means the later, finer axes are probed around a framing that is already
 * roughly right — a target offset solved against a wrong distance is solved
 * against the wrong parallax and has to be redone anyway.
 */
const AXES: readonly AxisSpec[] = [
  { axis: "distance_m", span: 1.6, minimum: 0.5, maximum: 5.0 },
  { axis: "azimuth_rad", span: 1.2 },
  { axis: "elevation_rad", span: 0.6, minimum: 0.02, maximum: 1.4 },
  { axis: "targetY", span: 0.5 },
  { axis: "targetX", span: 0.5 },
  { axis: "targetZ", span: 0.5 },
];

const PROBES_PER_AXIS = 7;

function read(camera: CameraState, axis: Axis): number {
  switch (axis) {
    case "targetX": return camera.target_m.x;
    case "targetY": return camera.target_m.y;
    case "targetZ": return camera.target_m.z;
    default: return camera[axis];
  }
}

function withAxis(camera: CameraState, axis: Axis, value: number): CameraState {
  switch (axis) {
    case "targetX": return { ...camera, target_m: { ...camera.target_m, x: value } };
    case "targetY": return { ...camera, target_m: { ...camera.target_m, y: value } };
    case "targetZ": return { ...camera, target_m: { ...camera.target_m, z: value } };
    default: return { ...camera, [axis]: value };
  }
}

/** Render a batch of framings in one process and score each against the plate. */
function scoreBatch(candidates: ReadonlyArray<{ label: string; camera: CameraState }>): Map<string, number> {
  const sweepDirectory = path.resolve(outDirectory, "sweep");
  const sweepFile = path.resolve(outDirectory, "sweep.json");
  writeFileSync(sweepFile, JSON.stringify(candidates.map((entry) => ({ label: entry.label, camera: entry.camera }))));
  const result = spawnSync(
    process.execPath,
    ["--import", "tsx", "tools/run-webgpu-exclusive.ts", "--import", "tsx", "tools/run-svo-dry-render-smoke.ts"],
    {
      stdio: ["ignore", "pipe", "inherit"],
      encoding: "utf8",
      env: {
        ...environment,
        WEBGPU_NODE_MODULE: environment.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`,
        FLUID_SVO_DRY_SMOKE_SCENE: "hero-garden-hose",
        FLUID_SVO_DRY_SMOKE_WIDTH: String(width),
        FLUID_SVO_DRY_SMOKE_HEIGHT: String(height),
        FLUID_SVO_DRY_SMOKE_WARMUPS: "1",
        FLUID_SVO_DRY_SMOKE_FRAMES: "1",
        FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP: sweepFile,
        FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP_DIR: sweepDirectory,
        FLUID_SVO_DRY_SMOKE_OUT: path.resolve(outDirectory, "smoke.json"),
      },
    },
  );
  if (result.status !== 0) {
    process.stdout.write(result.stdout ?? "");
    throw new Error(`camera sweep failed with status ${result.status}`);
  }
  const scores = new Map<string, number>();
  for (const candidate of candidates) {
    const frame = linearFromSrgbImage(decodeRgbPng(readFileSync(path.join(sweepDirectory, `${candidate.label}.png`))));
    const registered = registerToScoringGrid(frame, plate);
    scores.set(candidate.label, edgeAlignment(registered.frame, registered.plate));
  }
  return scores;
}

const start: CameraState = {
  ...defaultCamera,
  ...heroGardenCamera,
  ...(environment.FLUID_HERO_SOLVE_START ? JSON.parse(environment.FLUID_HERO_SOLVE_START) : {}),
  target_m: { ...(heroGardenCamera.target_m ?? defaultCamera.target_m) },
};

let best = start;
let bestScore = scoreBatch([{ label: "start", camera: start }]).get("start") ?? -1;
process.stdout.write(`[camera-solve] start alignment ${bestScore.toFixed(4)} at ${JSON.stringify(best)}\n`);

const trace: Array<Record<string, unknown>> = [
  { pass: 0, axis: "start", alignment: bestScore, camera: best },
];

for (let pass = 0; pass < passes; pass += 1) {
  // Halve the probe width each pass: the first pass finds the basin, later
  // passes find the bottom of it.
  const shrink = 0.5 ** pass;
  for (const spec of AXES) {
    const centre = read(best, spec.axis);
    const span = spec.span * shrink;
    const candidates: Array<{ label: string; camera: CameraState }> = [];
    for (let probe = 0; probe < PROBES_PER_AXIS; probe += 1) {
      const t = probe / (PROBES_PER_AXIS - 1) - 0.5;
      let value = centre + t * span;
      if (spec.minimum !== undefined) value = Math.max(spec.minimum, value);
      if (spec.maximum !== undefined) value = Math.min(spec.maximum, value);
      candidates.push({ label: `p${pass}-${spec.axis}-${probe}`, camera: withAxis(best, spec.axis, value) });
    }
    const scores = scoreBatch(candidates);
    let localBest = best;
    let localScore = bestScore;
    for (const candidate of candidates) {
      const score = scores.get(candidate.label) ?? -1;
      if (score > localScore) {
        localScore = score;
        localBest = candidate.camera;
      }
    }
    const moved = localScore > bestScore;
    best = localBest;
    bestScore = localScore;
    process.stdout.write(
      `[camera-solve] pass ${pass} ${spec.axis.padEnd(15)} ${moved ? "->" : "  "} ${read(best, spec.axis).toFixed(4)}` +
        `  alignment ${bestScore.toFixed(4)}\n`,
    );
    trace.push({ pass, axis: spec.axis, alignment: bestScore, camera: best });
  }
}

writeFileSync(
  path.join(outDirectory, "solution.json"),
  `${JSON.stringify({ plate: platePath, grid: { width, height }, alignment: bestScore, camera: best, trace }, null, 2)}\n`,
);
process.stdout.write(`\n[camera-solve] best alignment ${bestScore.toFixed(4)}\n`);
process.stdout.write(`[camera-solve] camera ${JSON.stringify(best, null, 2)}\n`);
process.stdout.write(`[camera-solve] solution written to ${path.join(outDirectory, "solution.json")}\n`);
