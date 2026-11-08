/**
 * Scene contact for the cloth system: lane C2's OGC detector attached to each
 * document's Tier A solver. It sees the cloth itself (self contact) and the
 * scene's solid world, through C2's adapter (`contact/scene-solids.ts`): the
 * container shell, box fills and the procedural bake on the scene's own voxel
 * lattice, in the solver's cloth frame.
 *
 * The solid field is the expensive part (seconds of CPU for a Bridson scene),
 * so it is built once per solid world and cloth frame and kept across cloth
 * rebuilds (a cut keeps the document's frame). A solid world the adapter
 * cannot follow, terrain or rigid bodies under cloth, a field beyond the
 * capacity the detector was built with, and every failing status bit halt
 * the scene explicitly; nothing falls back to contact-free stepping.
 */
import type { SceneDescription } from '../core/model';
import { solidWorldContentStamp } from '../core/solid-world';
import { OgcContactDetector, type OgcContactConfig } from './contact/ogc-detector';
import { buildOgcTopology, type OgcMeshTopology } from './contact/ogc-topology';
import { OGC_COUNTER, OGC_STATUS } from './contact/ogc-wgsl';
import { clothSolidBand, clothSolidSourceForScene } from './contact/scene-solids';
import { SOLID_SAMPLE_WORDS, buildClothSolidField, clothSolidSamples, type ClothSolidField } from './contact/solid-field';
import type { ClothSolverModel } from './oracle';
import type { ClothDocument, ClothVector } from './types';

/**
 * The detector's authored settings: C2's M2 gate values. The pair slots per
 * vertex cover the gates' measured peaks (809 N in the drop pile-up, 776 N in
 * the twist); the grid holds 32 entries per primitive. An overflow of any
 * capacity is a failing status bit, never a silent truncation.
 */
export const CLOTH_SCENE_CONTACT = {
  kc_N_m: 100,
  epsV_m_s: 1e-2,
  rqBase_m: 3e-3,
  rqMax_m: 0.04,
  envelopeScale: 2.5,
  contactCapacity: 128,
  gridEntriesPerPrimitive: 32,
  bucketCount: 1 << 18,
  roundCap: 4,
  pairSlotsPerVertex: 840,
} as const;

/** Status bits that halt the scene (as the M2 gates fail them). Envelope and round caps, and held pairs, are reported, not failures. */
export const CLOTH_CONTACT_FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow
  | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration | OGC_STATUS.solidBounds;

/** Byte offset of the detector's status word in its counter buffer. */
export const CLOTH_CONTACT_STATUS_OFFSET = 4 * OGC_COUNTER.status;

/** OGC's offset radius: the thickest material of the document. */
export function clothContactRadius(document: ClothDocument): number {
  const r = Math.max(...document.materials.map(m => m.thickness_m));
  if (!(r > 0 && r < CLOTH_SCENE_CONTACT.rqBase_m)) throw new Error(`Cloth contact needs a material thickness in (0, ${CLOTH_SCENE_CONTACT.rqBase_m}) m, got ${r}`);
  return r;
}

/** What the field depends on: the solid world, the cloth frame and the band. */
export function clothContactFieldKey(scene: SceneDescription, origin: ClothVector, r: number): string {
  return `${solidWorldContentStamp(scene)}|${scene.solidBoundary?.patchesKey ?? ''}|${origin.join(',')}|${r}`;
}

/**
 * The field's band: r + 1.5 L rqMax with L at sqrt(3) (clothSolidBand), plus
 * 1e-4 relative on L. The bricks' L_max is an f32 gradient bound that rounds
 * past sqrt(3) (measured 1.7320512 against 1.7320508), and the detector's
 * check uses that L_max: without the margin a container shell alone fails it.
 */
export function clothContactBand(r: number): number {
  return clothSolidBand(r, CLOTH_SCENE_CONTACT.rqMax_m) + 1.5e-4 * Math.sqrt(3) * CLOTH_SCENE_CONTACT.rqMax_m;
}

/** The scene's solid world as the detector's static field, in the cloth frame at `origin`. Setup and solid edits only. */
export function buildClothSceneField(scene: SceneDescription, origin: ClothVector, r: number): ClothSolidField {
  if (scene.terrain) throw new Error('Cloth contact does not see terrain yet: a cloth scene cannot hold terrain');
  if (scene.rigidBodies.length) throw new Error('Cloth contact does not see rigid bodies yet: a cloth scene cannot hold them');
  return buildClothSolidField(clothSolidSourceForScene(scene, { band_m: clothContactBand(r), frameOrigin_m: [origin[0], origin[1], origin[2]] }));
}

/** The detector's mesh and cloth-frame rest pose from the solver model. */
function meshOf(model: ClothSolverModel, origin: ClothVector): { mesh: OgcMeshTopology; rest: Float32Array } {
  const t = model.topology, T = t.faceCount, triangles = new Uint32Array(4 * T), rest = new Float32Array(4 * t.vertexCount);
  for (let f = 0; f < T; f++) { triangles.set(t.faces.subarray(3 * f, 3 * f + 3), 4 * f); triangles[4 * f + 3] = t.faceMaterial[f]!; }
  for (let v = 0; v < t.vertexCount; v++) for (let c = 0; c < 3; c++) rest[4 * v + c] = t.data.restPositions_m[3 * v + c]! - origin[c]!;
  return { mesh: { vertexCount: t.vertexCount, triangles, edges: t.edges }, rest };
}

/**
 * A detector for one document, sized from its mesh and the field it will see
 * (the field's words and sharp-feature samples, exactly). The caller builds
 * the solver around `solverContact` and `stepHooks`, then `bind`s it and sets
 * the field.
 */
export async function createClothSceneContact(device: GPUDevice, model: ClothSolverModel, document: ClothDocument, origin: ClothVector,
  field: ClothSolidField, dt_s: number, gravity_m_s2: readonly [number, number, number]): Promise<OgcContactDetector> {
  const r = clothContactRadius(document), { mesh, rest } = meshOf(model, origin), C = CLOTH_SCENE_CONTACT;
  const { vertexCount: N, faceCount: T, edgeCount: E } = model.topology;
  const edge = buildOgcTopology(mesh, rest).layout.maxEdgeLength;
  const config: OgcContactConfig = {
    r, kc: C.kc_N_m, mu: document.solver?.selfFriction ?? 0, epsV: C.epsV_m_s, dt: dt_s, gravity: gravity_m_s2,
    rqBase: C.rqBase_m, rqMax: C.rqMax_m, envelopeScale: C.envelopeScale, prediction: 'guess',
    contactCapacity: C.contactCapacity, pairCapacity: C.pairSlotsPerVertex * N, gridCapacity: C.gridEntriesPerPrimitive * (N + T + E),
    bucketCount: C.bucketCount, roundCap: C.roundCap, solidCapacityWords: field.words.length,
    solidSampleCapacity: clothSolidSamples(field, { r, edge }).length / SOLID_SAMPLE_WORDS,
  };
  return OgcContactDetector.create(device, mesh, rest, config);
}
