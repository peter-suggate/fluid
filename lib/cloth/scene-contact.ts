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
 * cannot follow, terrain under cloth, a field beyond the capacity the
 * detector was built with, and every failing status bit halt the scene
 * explicitly; nothing falls back to contact-free stepping.
 *
 * A moving solid is a scene rigid body with a prescribed pose track (C2's M6,
 * one-way): its shape is baked once in its body frame (`buildClothBodyField`)
 * and the frame's owner uploads its poses at every step boundary of a frame
 * (`clothBodyPoses`). At most one; any other rigid body fails explicitly.
 */
import type { RigidBodyDescription, SceneDescription } from '../core/model';
import { sampleRigidPoseTrack } from '../core/prescribed-rigid-motion';
import { sceneCellSizes_m } from '../core/scene-lattice';
import { applySolidBoxCuts } from '../core/solid-cuts';
import { solidWorldContentStamp } from '../core/solid-world';
import { OGC_HULL_EDGES_PER_SAMPLE, OgcContactDetector, type OgcBodyPose, type OgcContactConfig } from './contact/ogc-detector';
import { buildOgcTopology, type OgcMeshTopology } from './contact/ogc-topology';
import { OGC_COUNTER, OGC_STATUS } from './contact/ogc-wgsl';
import { clothSolidBand, clothSolidSourceForScene } from './contact/scene-solids';
import { SOLID_SAMPLE_WORDS, buildClothSolidField, clothSolidSampleEdges, clothSolidSamples, type ClothSolidField } from './contact/solid-field';
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
  // IPC's friction regularization: below eps_v a sticking contact creeps at about eps_v-scale speeds. 1 mm/s (coordinator
  // ruling 2026-11-10): the 20 degree incline patch creeps 0.6 mm/s (4.8 mm/s at 1 cm/s), the 35 degree slide is unchanged.
  epsV_m_s: 1e-3,
  rqBase_m: 3e-3,
  rqMax_m: 0.04,
  envelopeScale: 2.5,
  contactCapacity: 128,
  gridEntriesPerPrimitive: 32,
  bucketCount: 1 << 18,
  // Re-detection rounds per step: at least the Newton iteration cap (CLOTH_NEWTON_SCENE_DEFAULTS.iterations, 8; a
  // document authoring more Newton iterations gets as many). A round can follow each Newton iteration's filter, so a
  // smaller cap ends a step's last iterations early: at 4 the ledge strip (C2's ladder) lagged its free twin's strain by
  // a tick, its vertices over the ledge's edge carried at most 4 solid certificates' reach. Rounds run only when a filter
  // bounds a vertex, so the cap costs nothing where the motion is certified.
  roundCap: 8,
  pairSlotsPerVertex: 840,
  // The contact filter's plane records per vertex unless the document authors its own (contactPlaneRecordsPerVertex).
  planeRecordsPerVertex: 16,
} as const;

/** Status bits that halt the scene (as the M2 and M6 gates fail them). Envelope and round caps, and held pairs, are reported, not failures. */
export const CLOTH_CONTACT_FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow
  | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration | OGC_STATUS.solidBounds
  | OGC_STATUS.bodySqueeze | OGC_STATUS.heldConflict | OGC_STATUS.bodyPose | OGC_STATUS.coincident;

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

/**
 * The scene's moving solids: rigid bodies whose pose is prescribed by a pose track (one-way: the cloth never moves
 * them). A body the cloth cannot see that way (dynamic, a motor, an orbit, a plugin's kinematic pose, a fixed rigid
 * body that is not a voxel solid) fails explicitly, as do more than one.
 */
export function clothSceneMovingSolids(scene: SceneDescription): readonly RigidBodyDescription[] {
  for (const b of scene.rigidBodies)
    if (!(b.motion === 'static' && b.poseTrack?.keyframes.length && !b.prescribedMotion && !b.motorAttachment && !b.kinematic))
      throw new Error(`Cloth contact sees a rigid body only as a moving solid with a prescribed pose track (one-way); '${b.id}' is not one`);
  if (scene.rigidBodies.length > 1) throw new Error(`Cloth contact takes at most one moving solid; the scene has ${scene.rigidBodies.length} (${scene.rigidBodies.map(b => b.id).join(', ')})`);
  return scene.rigidBodies;
}

/** Most poses one frame uploads for the moving solid: the most steps a frame takes plus one (system.ts asserts it). */
export const CLOTH_BODY_POSES_PER_FRAME = 5;

/**
 * The moving solid's shape baked in its body frame (body origin at 0, axes of the identity orientation) on the scene's
 * lattice cell, with the scene contact's band and the body's friction, minus the body's local cuts. Setup and edit
 * time only (ClothSystem.adopt rebakes a cut body). A shape it cannot bake fails explicitly.
 */
export function buildClothBodyField(scene: SceneDescription, body: RigidBodyDescription, r: number): ClothSolidField {
  const cell = sceneCellSizes_m(scene)[0], band = clothContactBand(r), cuts = body.cuts ?? [];
  switch (body.shape) {
    case 'sphere': {
      const R = body.dimensions_m.x, reach = R + band + 2 * cell;
      if (!(R > 0)) throw new Error(`Moving solid '${body.id}' needs a positive sphere radius`);
      // The body's local cuts are in this same body frame (the shared editing path stores them so): each one
      // subtracts an oriented box, max(d, -box), which stays 1-Lipschitz. The field's resolution check refuses
      // a remnant thinner than its cell.
      const sdf = (p: readonly number[]) => applySolidBoxCuts(Math.hypot(p[0]!, p[1]!, p[2]!) - R, cuts, { x: p[0]!, y: p[1]!, z: p[2]! });
      return buildClothSolidField({ parts: [{ kind: 'sdf', sdf, lo: [-R, -R, -R], hi: [R, R, R], mu: body.friction }],
        lo: [-reach, -reach, -reach], hi: [reach, reach, reach], cell, band });
    }
    default: throw new Error(`Cloth contact cannot bake a moving solid of shape ${body.shape} yet ('${body.id}')`);
  }
}

/** The moving solid's poses in the cloth frame at `origin`, at the step boundaries t0 + k dt, k = 0..steps. */
export function clothBodyPoses(body: RigidBodyDescription, origin: ClothVector, t0_s: number, steps: number, dt_s: number): OgcBodyPose[] {
  return Array.from({ length: steps + 1 }, (_, k) => {
    const { position_m: c, orientation: q } = sampleRigidPoseTrack(body.poseTrack!, t0_s + k * dt_s);
    return { position: [c.x - origin[0], c.y - origin[1], c.z - origin[2]], orientation: [q.x, q.y, q.z, q.w] };
  });
}

/** The scene's solid world as the detector's static field, in the cloth frame at `origin`. Setup and solid edits only. */
export function buildClothSceneField(scene: SceneDescription, origin: ClothVector, r: number): ClothSolidField {
  if (scene.terrain) throw new Error('Cloth contact does not see terrain yet: a cloth scene cannot hold terrain');
  clothSceneMovingSolids(scene);
  return buildClothSolidField(clothSolidSourceForScene(scene, { band_m: clothContactBand(r), frameOrigin_m: [origin[0], origin[1], origin[2]] }));
}

/**
 * A field's sample capacity, exactly: its sharp features' surface samples, or as many as its samples' hull edges need
 * (the detector stores up to OGC_HULL_EDGES_PER_SAMPLE edges per authored sample); a rounded body has many edges.
 */
function sampleCapacity(field: ClothSolidField, r: number, edge: number): number {
  const options = { r, edge };
  return Math.max(clothSolidSamples(field, options).length / SOLID_SAMPLE_WORDS,
    Math.ceil(clothSolidSampleEdges(field, options).length / 2 / OGC_HULL_EDGES_PER_SAMPLE));
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
 * (the fields' words, sharp-feature samples and their hull edges, exactly). The caller builds
 * the solver around `solverContact` and `stepHooks`, then `bind`s it and sets
 * the field.
 */
export async function createClothSceneContact(device: GPUDevice, model: ClothSolverModel, document: ClothDocument, origin: ClothVector,
  field: ClothSolidField, dt_s: number, gravity_m_s2: readonly [number, number, number], bodyField?: ClothSolidField, newton = false): Promise<OgcContactDetector> {
  const r = clothContactRadius(document), { mesh, rest } = meshOf(model, origin), C = CLOTH_SCENE_CONTACT;
  const { vertexCount: N, faceCount: T, edgeCount: E } = model.topology;
  const edge = buildOgcTopology(mesh, rest).layout.maxEdgeLength;
  const config: OgcContactConfig = {
    r, kc: C.kc_N_m, mu: document.solver?.selfFriction ?? 0, epsV: C.epsV_m_s, dt: dt_s, gravity: gravity_m_s2,
    rqBase: C.rqBase_m, rqMax: C.rqMax_m, envelopeScale: C.envelopeScale, prediction: 'target',
    contactCapacity: C.contactCapacity, pairCapacity: C.pairSlotsPerVertex * N, gridCapacity: C.gridEntriesPerPrimitive * (N + T + E),
    bucketCount: C.bucketCount, roundCap: Math.max(C.roundCap, document.solver?.newton?.iterations ?? 0), solidCapacityWords: field.words.length, newton,
    // E12 (iii): the envelope capped at gamma_r times the smallest rest altitude, so a vertex's records are its approaches.
    envelopeMeshCap: true,
    // INTERIM (demand bug, C2): the document's authored plane records per vertex, until the relative-motion pair test.
    ...(newton ? { planeRecordCapacity: (document.solver?.contactPlaneRecordsPerVertex ?? C.planeRecordsPerVertex) * N } : {}),
    solidSampleCapacity: sampleCapacity(field, r, edge),
    ...(bodyField ? { body: { capacityWords: bodyField.words.length, sampleCapacity: sampleCapacity(bodyField, r, edge),
      poseCapacity: CLOTH_BODY_POSES_PER_FRAME } } : {}),
  };
  return OgcContactDetector.create(device, mesh, rest, config);
}
