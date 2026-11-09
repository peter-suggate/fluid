import { createEmptyScene } from "../../../core/empty-scene";
import type { CameraState, RigidBodyDescription, SceneDescription, Vec3 } from "../../../core/model";
import { bakeProceduralSolid } from "../../../core/procedural-solid";
import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "../../../core/solid-world";
import { VOXEL_MATERIAL_IDS } from "../../../core/voxel-scene";
import type { ClothDocument, ClothPendingSolid } from "../../types";
import type { Bridson2002Scene } from "./scenes";
import { bridsonShapeBox, bridsonShapeReach, bridsonSolidDistance, bridsonSolidPose, type BridsonSolid } from "./solids";

/**
 * A Bridson 2002 scene as a product scene document: a dry scene (no fluid
 * solver) whose cloth runs on the scene clock, one implicit 1/60 s step per
 * frame, and whose solids are ordinary scene voxel solids that render,
 * hit-test and edit through the shared solid-world path.
 *
 * Solids that hold still at t = 0 (the ground, legs, frustum, and the curtain
 * ball and spinning sphere until they start to move) are baked at their t = 0
 * pose. A solid in motion is a scene rigid body with a prescribed pose.
 * With `movingSolids: "rigid-body"` (what the catalog opens) each solid with
 * a pose track is a scene rigid body carrying that track for the whole run,
 * which the cloth system's contact (lane C2, M6, one-way) bakes in its body
 * frame and poses every step; it renders, picks and takes local cuts through
 * the shared rigid surface path (lib/core/rigid-surface.ts). With
 * `movingSolids: "held"` (this function's default, which the contact lane's
 * scene-solid tests are written against) each is held voxels recorded on the
 * cloth document as a pending solid, and the cloth system fails explicitly
 * when the clock reaches it: at the first motion of a held solid, or when a
 * moving one first comes within reach of the cloth.
 *
 * Scene coordinates are the paper's translated by `bridson2002SceneLayout(s)
 * .translation_m`: the container is centred on x and z with its floor at
 * y = 0, and the paper's ground plane lies one brick above it.
 */
export interface Bridson2002SceneLayout {
  /** Scene position = paper position + translation. */
  readonly translation_m: Vec3;
  readonly cell_m: number;
  /** Container lattice, cells (whole bricks). */
  readonly dimensions: readonly [number, number, number];
  /** Paper position of lattice cell (0, 0, 0). */
  readonly lower_m: Vec3;
  /** Solids baked as voxel solids at their t = 0 pose. */
  readonly held: readonly string[];
  /** Solids the scene cannot represent yet, and when the cloth system stops for each. */
  readonly pending: readonly ClothPendingSolid[];
}

export interface Bridson2002SceneOptions {
  /** `bake: false` skips the voxel bake (library card previews). */
  readonly bake?: boolean;
  /** How a solid with a pose track enters the scene: held voxels then pending (default), or a scene rigid body (the catalog's choice). */
  readonly movingSolids?: "held" | "rigid-body";
}

const BRICK = 8;
const IDENTITY_TOLERANCE = 1e-12;
const ALIGNED_TOLERANCE_CELLS = 1e-6;

/** The first time a solid's pose departs from its t = 0 pose; Infinity when it never does. */
export function bridsonSolidMotionStart(solid: BridsonSolid): number {
  const m = solid.motion;
  if (m.kind === "static") return Infinity;
  const frames = m.track.keyframes, a = frames[0]!;
  for (let i = 1; i < frames.length; i++) {
    const b = frames[i]!, p = a.position_m, q = b.position_m, r = a.orientation, s = b.orientation;
    const moved = Math.hypot(q.x - p.x, q.y - p.y, q.z - p.z) > 0
      || Math.abs(Math.abs(r.w * s.w + r.x * s.x + r.y * s.y + r.z * s.z) - 1) > IDENTITY_TOLERANCE;
    if (moved) return Math.max(0, frames[i - 1]!.time_s);
  }
  return Infinity;
}

interface Box { lo: [number, number, number]; hi: [number, number, number] }
const emptyBox = (): Box => ({ lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity] });
function grow(box: Box, lo: readonly number[], hi: readonly number[]): void {
  for (let a = 0; a < 3; a++) { box.lo[a] = Math.min(box.lo[a]!, lo[a]!); box.hi[a] = Math.max(box.hi[a]!, hi[a]!); }
}

function poseBox(solid: BridsonSolid, time_s: number): Box {
  const { position_m: c, orientation: q } = bridsonSolidPose(solid, time_s), box = emptyBox();
  if (Math.abs(Math.abs(q.w) - 1) > IDENTITY_TOLERANCE) {
    const r = bridsonShapeReach(solid.shape);
    grow(box, [c.x - r, c.y - r, c.z - r], [c.x + r, c.y + r, c.z + r]);
  } else {
    const b = bridsonShapeBox(solid.shape);
    grow(box, [c.x - b.halfX, c.y + b.yMin, c.z - b.halfZ], [c.x + b.halfX, c.y + b.yMax, c.z + b.halfZ]);
  }
  return box;
}

function clothRestBox(source: Bridson2002Scene): Box {
  const rest = source.cloth.geometry.data.restPositions_m, box = emptyBox();
  for (let v = 0; v < rest.length; v += 3) grow(box, [rest[v]!, rest[v + 1]!, rest[v + 2]!], [rest[v]!, rest[v + 1]!, rest[v + 2]!]);
  return box;
}

/** First step time at which a solid moving from t = 0 would reach the cloth's rest bounds: the gate fires before that step. */
function firstReach(solid: BridsonSolid, cloth: Box, source: Bridson2002Scene): number {
  const r = bridsonShapeReach(solid.shape), steps = Math.ceil(source.duration_s / source.step_s - 1e-9);
  for (let n = 0; n <= steps; n++) {
    const t = n * source.step_s, c = bridsonSolidPose(solid, t).position_m;
    const d = Math.hypot(...[c.x, c.y, c.z].map((v, a) => Math.max(cloth.lo[a]! - v, 0, v - cloth.hi[a]!)));
    if (d <= r) return Math.max(0, (n - 1) * source.step_s);
  }
  return source.duration_s;
}

export function bridson2002SceneLayout(source: Bridson2002Scene): Bridson2002SceneLayout {
  const h = source.voxelCell_m, brick = BRICK * h, cloth = clothRestBox(source);
  const held: BridsonSolid[] = [], pending: ClothPendingSolid[] = [];
  for (const solid of source.solids) {
    const start = bridsonSolidMotionStart(solid);
    if (start > 0) held.push(solid);
    if (start === 0) pending.push({ id: solid.id, start_s: firstReach(solid, cloth, source) });
    else if (Number.isFinite(start)) pending.push({ id: solid.id, start_s: start });
  }
  // The container holds the cloth, every held solid, and every track a moving solid will sweep.
  const bounds = emptyBox();
  grow(bounds, cloth.lo, cloth.hi);
  for (const s of held) { const b = poseBox(s, 0); grow(bounds, b.lo, b.hi); }
  for (const s of source.solids) if (bridsonSolidMotionStart(s) < Infinity) {
    for (let t = 0; t <= source.duration_s + 1e-9; t += source.step_s) { const b = poseBox(s, t); grow(bounds, b.lo, b.hi); }
  }
  // The floor is the bottom of the lowest held box (the ground slab); a solid buried deeper is clipped there.
  const boxes = held.filter(s => s.shape.kind === "box");
  const floor = boxes.length ? Math.min(...boxes.map(s => poseBox(s, 0).lo[1])) : bounds.lo[1];
  const lowerCells = [0, 1, 2].map(a => a === 1 ? Math.floor(floor / brick + 1e-9) : Math.floor(bounds.lo[a]! / brick + 1e-9) - 1);
  const upperCells = [0, 1, 2].map(a => Math.ceil(bounds.hi[a]! / brick - 1e-9) + (a === 1 ? 2 : 1));
  const lower = lowerCells.map(v => v * brick), dimensions = [0, 1, 2].map(a => BRICK * (upperCells[a]! - lowerCells[a]!)) as unknown as [number, number, number];
  const width = dimensions[0] * h, depth = dimensions[2] * h;
  return {
    translation_m: { x: -lower[0]! - width / 2, y: -lower[1]!, z: -lower[2]! - depth / 2 },
    cell_m: h, dimensions, lower_m: { x: lower[0]!, y: lower[1]!, z: lower[2]! },
    held: held.map(s => s.id), pending,
  };
}

/** Held solids as voxel patches: lattice-aligned boxes as exact fills, every other shape through one procedural bake. */
function bakeHeldSolids(source: Bridson2002Scene, layout: Bridson2002SceneLayout, rigid: ReadonlySet<string>): Pick<SceneDescription, "solidVoxels" | "solidBoundary"> {
  const h = layout.cell_m, o = layout.lower_m, dims = layout.dimensions;
  const held = source.solids.filter(s => layout.held.includes(s.id) && !rigid.has(s.id));
  const fills: SolidWorldVoxelPatch[] = [], shaped: BridsonSolid[] = [];
  for (const solid of held) {
    const pose = bridsonSolidPose(solid, 0), box = poseBox(solid, 0);
    const cells = [...box.lo.map((v, a) => (v - [o.x, o.y, o.z][a]!) / h), ...box.hi.map((v, a) => (v - [o.x, o.y, o.z][a]!) / h)];
    const aligned = solid.shape.kind === "box" && Math.abs(Math.abs(pose.orientation.w) - 1) <= IDENTITY_TOLERANCE
      && cells.every(c => Math.abs(c - Math.round(c)) < ALIGNED_TOLERANCE_CELLS);
    if (!aligned) { shaped.push(solid); continue; }
    const r = cells.map(Math.round), clip = (v: number, a: number) => Math.max(0, Math.min(dims[a]!, v));
    fills.push({ operation: "fill", minimum: [clip(r[0]!, 0), clip(r[1]!, 1), clip(r[2]!, 2)],
      maximumExclusive: [clip(r[3]!, 0), clip(r[4]!, 1), clip(r[5]!, 2)], materialId: VOXEL_MATERIAL_IDS.terrain });
  }
  if (!shaped.length) return { solidVoxels: fills };
  // A brick-aligned lattice around the shaped solids with a two-cell margin, clipped to the container.
  const bounds = emptyBox();
  for (const s of shaped) { const b = poseBox(s, 0); grow(bounds, b.lo, b.hi); }
  const origin = [o.x, o.y, o.z];
  const start = [0, 1, 2].map(a => Math.max(0, BRICK * Math.floor(((bounds.lo[a]! - 2 * h) - origin[a]!) / (BRICK * h) + 1e-9)));
  const end = [0, 1, 2].map(a => Math.min(dims[a]!, BRICK * Math.ceil(((bounds.hi[a]! + 2 * h) - origin[a]!) / (BRICK * h) - 1e-9)));
  const size = [0, 1, 2].map(a => end[a]! - start[a]!) as unknown as [number, number, number];
  const baked = bakeProceduralSolid(size, [h, h, h], q => {
    const p = { x: origin[0]! + (start[0]! + q[0]) * h, y: origin[1]! + (start[1]! + q[1]) * h, z: origin[2]! + (start[2]! + q[2]) * h };
    let d = Infinity;
    for (const s of shaped) d = Math.min(d, bridsonSolidDistance(s, p, 0));
    return d / h;
  }, VOXEL_MATERIAL_IDS.white);
  const shift = <T extends readonly number[]>(v: T) => v.map((x, a) => x + start[a]!) as unknown as [number, number, number];
  const patches = baked.solidVoxels.map(p => ({ ...p, minimum: shift(p.minimum), maximumExclusive: shift(p.maximumExclusive) }));
  // The procedural patches come first: a later fill owns the cells it covers, so the slab and a solid buried in it union exactly.
  return {
    solidVoxels: [...patches, ...fills],
    solidBoundary: { ...baked.solidBoundary, dimensions: [...dims], patchesKey: solidVoxelPatchesKey(patches),
      bricks: baked.solidBoundary.bricks.map(b => ({ ...b, origin: shift(b.origin) })) },
  };
}

/** The moving solids as scene rigid bodies: the pose track in scene coordinates, one-way (the cloth never moves them). */
function movingRigidBodies(source: Bridson2002Scene, t: Vec3): RigidBodyDescription[] {
  return source.solids.filter(s => bridsonSolidMotionStart(s) < Infinity).map(solid => {
    if (solid.motion.kind !== "pose-track") throw new Error(`Bridson 2002 solid '${solid.id}' moves without a pose track`);
    if (solid.shape.kind !== "sphere") throw new Error(`Bridson 2002 moving solid '${solid.id}' is a ${solid.shape.kind}; scene rigid bodies take its shape only as a sphere`);
    const keyframes = solid.motion.track.keyframes.map(k => ({ time_s: k.time_s, orientation: { ...k.orientation },
      position_m: { x: k.position_m.x + t.x, y: k.position_m.y + t.y, z: k.position_m.z + t.z } }));
    const pose = bridsonSolidPose(solid, 0), r = solid.shape.radius_m;
    return { id: solid.id, name: solid.id, shape: "sphere", dimensions_m: { x: r, y: r, z: r }, density_kg_m3: 1000,
      position_m: { x: pose.position_m.x + t.x, y: pose.position_m.y + t.y, z: pose.position_m.z + t.z }, orientation: { ...pose.orientation },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0, friction: solid.friction,
      motion: "static", poseTrack: { keyframes, ...(solid.motion.track.localOffset_m ? { localOffset_m: { ...solid.motion.track.localOffset_m } } : {}) } };
  });
}

/** The document the product opens (see Bridson2002SceneOptions). */
export function bridson2002SceneDescription(source: Bridson2002Scene, options: Bridson2002SceneOptions = {}): SceneDescription {
  const layout = bridson2002SceneLayout(source), h = layout.cell_m, t = layout.translation_m;
  const rigidBodies = options.movingSolids === "rigid-body" ? movingRigidBodies(source, t) : [];
  const rigid = new Set(rigidBodies.map(b => b.id)), pending = layout.pending.filter(p => !rigid.has(p.id));
  const scene = createEmptyScene({ sceneId: source.id, finestCellSize_m: h,
    extents_m: { x: layout.dimensions[0] * h, y: layout.dimensions[1] * h, z: layout.dimensions[2] * h } });
  scene.duration_s = source.duration_s;
  scene.numerics = { ...scene.numerics, fixedDt_s: source.step_s, maxDt_s: source.step_s };
  if (options.bake !== false) {
    const solids = bakeHeldSolids(source, layout, rigid);
    // The procedural bake's patches lead: its boundary covers the prefix solidVoxels.slice(0, patchCount).
    scene.solidVoxels = [...solids.solidVoxels, ...scene.solidVoxels];
    if (solids.solidBoundary) scene.solidBoundary = solids.solidBoundary;
  }
  const data = source.cloth.geometry.data, rest = data.restPositions_m.map((v, i) => v + [t.x, t.y, t.z][i % 3]!);
  const vertexCount = rest.length / 3;
  const cloth: ClothDocument = { ...source.cloth,
    geometry: { kind: "inline", data: { ...data, restPositions_m: rest, vertexMass_kg: new Array(vertexCount).fill(source.nodeMass_kg) } },
    ...(pending.length ? { pendingSolids: pending } : {}) };
  scene.cloth = [cloth];
  scene.rigidBodies = rigidBodies;
  return scene;
}

/** Frames the cloth's rest sheet from above and to the side. */
export function bridson2002Camera(scene: SceneDescription): Partial<CameraState> {
  const rest = scene.cloth?.[0]?.geometry.data.restPositions_m;
  if (!rest?.length) return {};
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < rest.length; i++) { lo[i % 3] = Math.min(lo[i % 3]!, rest[i]!); hi[i % 3] = Math.max(hi[i % 3]!, rest[i]!); }
  const span = Math.max(hi[0]! - lo[0]!, hi[2]! - lo[2]!);
  return { azimuth_rad: 0.72, elevation_rad: 0.5, distance_m: 2.2 * span,
    target_m: { x: (lo[0]! + hi[0]!) / 2, y: 0.6 * hi[1]!, z: (lo[2]! + hi[2]!) / 2 } };
}
