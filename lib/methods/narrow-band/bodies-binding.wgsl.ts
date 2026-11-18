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
  /** The rigid exchange (the rigid system's): couple's words, GPU_RIGID_EXCHANGE_WORDS i32 a body. Zero before each
   * step's couple: the host clears it after the integrator reads it (integrate), and it starts zeroed. */
  readonly exchange: GPUBuffer;
  /** The fluid density (kg/m^3): couple's p = X rho / dt. */
  readonly density_kg_m3: number;
  /** A4: the roster into the rigid state buffer through the queue (prescribed motion, roster changes; syncBodies
   * writes only on a change), so that the step's part 1 sees it; returns the step's header. Called once with
   * step -1 before the boot (the t = 0 roster, as the old host's construction syncs it), then before each step's
   * ring write, after step `step - 1`'s part 2 and integrator are queued: the roster sampled at the step's end,
   * (step + 1) dt, as the old advance samples it. Once a step. */
  sync(step: number): BrickBodiesHeader;
  /** §7.1: step `step`'s rigid integration, submitted with its part 2 (after couple): rigidSystem.encode's tables and
   * integrate, then the exchange's clear. Undefined when the coupling is off (the old host's uncoupled bodies: no
   * couple and no integration). */
  integrate(step: number): GPUCommandBuffer | undefined;
}

/** The stamp's brick candidates (bodyStamp's BMASK, §3): the bodies within reach of the brick at cell origin o, the
 * reach length(2h), one cell and the swept reach (|v| + |w| r) horizon: their bounding sphere meets the brick's box,
 * and their own distance (pbBodyDistance, the mark's test) at the brick's middle, less its half-diagonal, is within the
 * reach. The census's body rule reads the same test, so every brick the stamp gives a K slot is held h. Needs
 * pbBrickBodiesWGSL and `bodies`. */
export const PB_BRICK_CANDIDATES_WGSL = /* wgsl */`
// The brick's BMASK (§3): pbBodyBoxCandidates with each body's own reach, length(2h) + one cell + its swept reach
// (|v| + |w| r) horizon: the mark's margin, so every body setMark marked the brick for is a candidate (a marked tile's
// middle lies within length(2h) of the brick's, so its distance less its own half-diagonal is at least the brick's
// less length(4h)). The sphere alone took a long body's whole sphere: beach's wave piston, a 1 x 4 x 24 m box, a
// 12.2 m sphere and 2705 dry bricks NEAR_BODY at its boot, their K slots and the h hold (2026-11-18).
fn pbBrickCandidates(o:vec3i)->u32{
 let h=pbH();let lo=pbWorldCell(o)-0.5*h;let hi=lo+8.0*h;let cell=max(h.x,max(h.y,h.z));let dt=pbBrickHorizon();
 let middle=0.5*(lo+hi);let half=length(4.0*h);
 var mask=0u;
 for(var i=0u;i<pbBodyCount();i++){
  let b=bodies[i];let c=b.positionShape.xyz;let radius=max(b.dimensions.w,0.0);
  let reach=length(2.0*h)+cell+(length(b.linearVelocity.xyz)+length(b.angularVelocity.xyz)*radius)*dt;
  if(b.positionShape.w<0.0){mask|=1u<<i;continue;}
  if(distance(clamp(c,lo,hi),c)>radius+reach){continue;}
  if(pbBodyDistance(i,middle)-half<=reach){mask|=1u<<i;}
 }
 return mask;
}
`;
