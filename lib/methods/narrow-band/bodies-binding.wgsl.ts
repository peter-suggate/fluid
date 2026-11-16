/** Lane P's body bindings for brick kernels (M5 design note §7.2, A3): the `bodies` and `bodyCuts` bindings, the
 * ring's body header readers and gate, the shared pose queries, and setMark's body test (A2). No brick-set import, so
 * E's brick-set.wgsl.ts can take it (bodies-stage.wgsl.ts holds the stamp). */
import type { BrickBinding } from "./brick-kernel";
import { pbBodyShapesWGSL, type PbBodySource } from "./bodies-shapes.wgsl";

export type Vec3 = readonly [number, number, number];

/** Where the lattice sits in the world, for the body queries (world metres). */
export interface BrickBodiesFrame {
  /** The world position (m) of lattice cell (0,0,0)'s low corner (BrickSceneSetup.origin_m). */
  readonly frame_m: Vec3;
  /** The cell size (m). */
  readonly h_m: number;
}

export const BODIES_RO: BrickBinding = { name: "bodies", kind: "ro", type: "array<PBRigidBody>" };
export const BODY_CUTS_RO: BrickBinding = { name: "bodyCuts", kind: "ro", type: "array<vec4f>" };

const f32 = (v: number) => { const s = Math.fround(v).toPrecision(9); return /[.e]/.test(s) ? s : `${s}.0`; };
export const pbWgslVec3 = (v: Vec3) => `vec3f(${v.map(f32).join(",")})`;

/** The shared module's source in a brick kernel that binds `bodies`, `bodyCuts` and `cfg` (the ring slot). */
export function pbBrickBodySource(frame: BrickBodiesFrame): PbBodySource {
  return { bodies: "bodies", count: "pbBrickBodyCount()", cuts: "bodyCuts", frame: pbWgslVec3(frame.frame_m), h: pbWgslVec3([frame.h_m, frame.h_m, frame.h_m]), cutFail: "pbBrickCutFail" };
}

/** The ring gate and header readers, and the shared pose queries, for a brick kernel (needs the ABI and `cfg`). */
export function pbBrickBodiesWGSL(frame: BrickBodiesFrame): string {
  return /* wgsl */`
fn pbBrickBodiesOn()->bool{return (nbRing(cfg.x,NB_RING_FLAGS)&NB_RING_FLAG_BODIES)!=0u;}
fn pbBrickBodyCount()->u32{return select(0u,nbRing(cfg.x,NB_RING_BODY_COUNT),pbBrickBodiesOn());}
fn pbBrickHorizon()->f32{return nbRingF(cfg.x,NB_RING_BODY_HORIZON);}
fn pbBrickCutFail(i:u32){nbFail(NB_FAIL_UNSUPPORTED,0xb2000000u|i);}
${pbBodyShapesWGSL(pbBrickBodySource(frame))}`;
}

/** setMark's body test (A2, §4): whether any body may touch tile t (0-7, x fastest) of the brick at cell origin o
 * within the stamp horizon (pbBodyMarkTile). False when the ring's BODIES gate is off. */
export const PB_BRICK_MARK_WGSL = /* wgsl */`
fn pbBrickMarkTile(o:vec3i,t:u32)->bool{
 if(!pbBrickBodiesOn()){return false;}
 let c=o+4*vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));
 // A tile past the lattice (a skirt brick's: the vertex row and the plane-N walls) is no body tile, as the old mark
 // takes live tiles only (umBodyMarkTile's umTileLive): it holds no rows, and a K slot there latches the solver.
 if(!nbInLattice(c)){return false;}
 let low=pbWorldCell(c)-0.5*pbH();
 return pbBodyMarkTile(low,pbBrickHorizon());
}`;


/** The ring's body header for one step (A3): RING.BODY_COUNT, BODY_FLAGS (BODY_FLAG bits) and BODY_GEN. */
export interface BrickBodiesHeader { readonly count: number; readonly flags: number; readonly generation: number }

/** The scene's rigid bodies as the brick engine binds them (A3, A4): the rigid state buffer and the cut atlas (the
 * rigid system's own buffers: the engine binds them, never allocates, saves or destroys them), the frame, and the
 * host's per-step sync. */
export interface BrickBodiesBinding extends BrickBodiesFrame {
  /** The container's top (the static box rule the stamp reads when the scene binds no voxel mask). */
  readonly openTop: boolean;
  /** The rigid state buffer: 24 records of 128 B (webgpu-rigid-body.ts:73-92). */
  readonly bodies: GPUBuffer;
  /** The body-local cut atlas (packRigidBodyCutAtlas), written only after pbCheckCutAtlas. */
  readonly cuts: GPUBuffer;
  /** A4: called before step `step`'s ring write, after step `step - 1`'s part 2 is queued (and once before the boot,
   * step 0): syncBodies (prescribed motion, roster changes; only on change) into the state buffer through the queue,
   * so the step's part 1 sees it. Returns the step's header. Idempotent for a step: the engine calls sync(0) before
   * the boot and again before step 0's part 1, and the second call must change nothing (syncBodies's
   * unchangedSinceSync holds it). */
  sync(step: number): BrickBodiesHeader;
}
