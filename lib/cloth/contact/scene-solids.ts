/**
 * The scene's solid world as cloth contact sees it: a `ClothSolidSource` in
 * scene metres on the scene's own voxel lattice, so the contact field's nodes
 * are the nodes the scene renders, hit-tests and edits its solids on.
 *
 * Every authored solid patch is accounted for, or the build fails:
 *   - a box fill (the container shell, a ground slab) is a "lattice-box" part,
 *     exact under the trilinear field;
 *   - the patches of the scene's procedural bake (`solidBoundary`) are covered
 *     by the analytic solids the scene author sampled them from, as "sdf"
 *     parts; the build proves it by re-evaluating every stored boundary node;
 *   - any other patch (a clear, a paint, a sculpted or round kind, a voxel
 *     block from an edit) fails explicitly: cloth contact does not follow
 *     solid edits yet, and must not silently disagree with the visible solid.
 *
 * The field box is the container lattice; the cloth must stay inside it
 * (leaving it raises OGC_ST_SOLID_BOUNDS).
 */
import type { SceneDescription } from "../../core/model";
import type { ProceduralSolidBoundary } from "../../core/procedural-solid";
import { sceneCellSizes_m, sceneLatticeDimensions, solidVoxelShellForScene } from "../../core/scene-lattice";
import { SOLID_PATCH_BOX_KIND } from "../../core/solid-patch/kind";
import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "../../core/solid-world";
import { bridson2002ClothSceneSolids } from "../scenes/bridson-2002/contact-solids";
import type { ClothSolidPart, ClothSolidSource, V3 } from "./solid-field";

/**
 * Friction of the container shell. The container is the fluid's vessel; no
 * scene authors cloth friction for it, so its walls are frictionless.
 */
export const CLOTH_CONTAINER_SHELL_FRICTION = 0;

/** What a scene author knows about its solids that the voxel world does not carry. */
export interface ClothSceneSolidAuthoring {
  /** The analytic solids the scene's procedural bake was sampled from, scene metres, each with its friction. */
  readonly procedural?: readonly ClothSolidPart[];
  /** Friction of an authored box fill other than the container shell; throws for a box the author did not make. */
  boxFriction(patch: SolidWorldVoxelPatch, box: { lo: V3; hi: V3 }): number;
}

/** Without authoring a scene may hold only its container shell. */
const SHELL_ONLY: ClothSceneSolidAuthoring = {
  boxFriction(patch) {
    throw new Error(`Cloth contact: scene solid box ${patch.minimum}..${patch.maximumExclusive} has no authored cloth friction (author the scene's cloth solids in clothSolidSourceForScene)`);
  },
};

/** The band a detector needs: r + 1.5 L rqMax with L at its bound sqrt(3) for a lattice-sampled 1-Lipschitz solid. */
export function clothSolidBand(r_m: number, rqMax_m: number): number {
  return r_m + 1.5 * Math.sqrt(3) * rqMax_m;
}

export interface ClothSceneSolidOptions {
  /** Surface band (m), at least the detector's r + 1.5 L_max rqMax (clothSolidBand). */
  readonly band_m: number;
  /** Override the scene family's authoring (tests). */
  readonly authoring?: ClothSceneSolidAuthoring;
  /** Scene position of the cloth frame (ClothGPUSolver's origin_m): the source comes back in that frame. Default 0. */
  readonly frameOrigin_m?: V3;
}

/** A source re-expressed in a frame whose origin sits at scene position o. */
export function clothSolidSourceInFrame(src: ClothSolidSource, o: V3): ClothSolidSource {
  const shift = (v: V3): V3 => [v[0] - o[0], v[1] - o[1], v[2] - o[2]];
  const parts = src.parts?.map((part): ClothSolidPart => part.kind === "sdf"
    ? { kind: "sdf", lo: shift(part.lo), hi: shift(part.hi), mu: part.mu, sdf: p => part.sdf([p[0] + o[0], p[1] + o[1], p[2] + o[2]]) }
    : { ...part, lo: shift(part.lo), hi: shift(part.hi) });
  if (src.sdf) throw new Error("Cloth contact: only a source of parts can be moved to the cloth frame");
  return { ...src, ...(parts ? { parts } : {}), lo: shift(src.lo), hi: shift(src.hi), ...(src.origin ? { origin: shift(src.origin) } : {}) };
}

const isBoxFill = (p: SolidWorldVoxelPatch) => p.operation === "fill" && (p.kind ?? SOLID_PATCH_BOX_KIND) === SOLID_PATCH_BOX_KIND;
const patchId = (p: SolidWorldVoxelPatch) => `${p.operation}:${p.kind ?? SOLID_PATCH_BOX_KIND}:${p.materialId}:${p.minimum}:${p.maximumExclusive}`;

/** World position of lattice node (0, 0, 0): the solid world's origin. */
export function sceneLatticeOrigin_m(scene: SceneDescription): V3 {
  return [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
}

/** The indices [start, end) of the procedural bake's patches in solidVoxels: the boundary's own prefix, or the same run after a prepended shell. */
function proceduralRun(scene: SceneDescription, shellCount: number): [number, number] | undefined {
  const b = scene.solidBoundary;
  if (!b) return undefined;
  for (let s = 0; s <= shellCount && s + b.patchCount <= scene.solidVoxels.length; s++)
    if (solidVoxelPatchesKey(scene.solidVoxels.slice(s, s + b.patchCount)) === b.patchesKey) return [s, s + b.patchCount];
  throw new Error("Cloth contact: the scene's procedural solid boundary no longer matches its patches (an edit replaced the baked solid)");
}

/** Every stored boundary node equals the union of the analytic parts there (the bake stores phi / h as f32). */
function checkProcedural(boundary: ProceduralSolidBoundary, parts: readonly ClothSolidPart[], origin: V3, h: number): void {
  const N = 9, tol = 1e-5;
  const p: [number, number, number] = [0, 0, 0];
  for (const brick of boundary.bricks) for (let i = 0; i < N ** 3; i++) {
    const q = [i % N, Math.floor(i / N) % N, Math.floor(i / (N * N))];
    for (let a = 0; a < 3; a++) p[a] = origin[a]! + (brick.origin[a]! + q[a]!) * h;
    let d = Infinity;
    for (const part of parts) if (part.kind === "sdf") d = Math.min(d, part.sdf(p));
    const stored = brick.values[i]!;
    if (!(Math.abs(Math.fround(d / h) - stored) <= tol * Math.max(1, Math.abs(stored))))
      throw new Error(`Cloth contact: the scene's procedural solid at (${p.map(v => v.toFixed(4)).join(", ")}) reads ${stored} cells, its authored solids ${d / h}`);
  }
}

/** The scene's solids as a cloth solid source, from explicit authoring. */
export function clothSolidSourceFromScene(scene: SceneDescription, authoring: ClothSceneSolidAuthoring, band_m: number): ClothSolidSource {
  if (scene.container.shape === "sphere") throw new Error("Cloth contact: a spherical container's shell is not a lattice box (unsupported)");
  const cells = sceneCellSizes_m(scene), h = cells[0];
  if (!(Math.abs(cells[1] - h) <= 1e-12 * h && Math.abs(cells[2] - h) <= 1e-12 * h)) throw new Error(`Cloth contact needs a cubic scene lattice, got ${cells.join(" x ")} m`);
  const origin = sceneLatticeOrigin_m(scene), dims = sceneLatticeDimensions(scene);
  const at = (c: readonly number[]): V3 => [origin[0] + c[0]! * h, origin[1] + c[1]! * h, origin[2] + c[2]! * h];
  const shell = solidVoxelShellForScene(scene), shellIds = new Set(shell.map(patchId));
  const run = proceduralRun(scene, shell.length);
  const parts: ClothSolidPart[] = [];
  scene.solidVoxels.forEach((patch, i) => {
    const box = { lo: at(patch.minimum), hi: at(patch.maximumExclusive) };
    if (shellIds.has(patchId(patch))) { parts.push({ kind: "lattice-box", ...box, mu: CLOTH_CONTAINER_SHELL_FRICTION }); return; }
    if (run && i >= run[0] && i < run[1]) {
      if (!authoring.procedural?.length) throw new Error("Cloth contact: the scene has a procedural solid but no authored analytic solids for it");
      return;
    }
    if (isBoxFill(patch)) { parts.push({ kind: "lattice-box", ...box, mu: authoring.boxFriction(patch, box) }); return; }
    throw new Error(`Cloth contact cannot follow solid patch ${i} (${patch.operation} ${patch.kind ?? SOLID_PATCH_BOX_KIND}) yet: solid edits under cloth are not supported`);
  });
  if (run) {
    checkProcedural(scene.solidBoundary!, authoring.procedural!, origin, h);
    parts.push(...authoring.procedural!);
  }
  return { parts, lo: origin, hi: at(dims), origin, cell: h, band: band_m };
}

/** The scene's solids as a cloth solid source, by the scene family's authoring. */
export function clothSolidSourceForScene(scene: SceneDescription, options: ClothSceneSolidOptions): ClothSolidSource {
  const authoring = options.authoring ?? (scene.sceneId.startsWith("bridson-2002-") ? bridson2002ClothSceneSolids(scene) : SHELL_ONLY);
  const src = clothSolidSourceFromScene(scene, authoring, options.band_m);
  return options.frameOrigin_m ? clothSolidSourceInFrame(src, options.frameOrigin_m) : src;
}
