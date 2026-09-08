import type { SparseCM12CurrentMapLayout } from "./sparse-cm12-current-map.wgsl";

/** The final even sweep writes back into the immutable cache. */
export const SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS = 64;

/** Temporary use of existing map arenas, before map-node advection begins.
 * No additional storage is allocated. The original-valid mask occupies the
 * first component of each nodal vec3; the scratch vec3 plane is the second
 * Jacobi bank. Map-node evolution and filtering may reuse both afterwards.
 */
export interface SparseCM12CurrentMapVelocityLayout {
  readonly dimensions: readonly [number, number, number];
  readonly nodeCount: number;
  readonly velocityBaseWords: number;
  readonly scratchBaseWords: number;
  readonly fixedMaskBaseWords: number;
}

export function createSparseCM12CurrentMapVelocityLayout(
  map: SparseCM12CurrentMapLayout,
): SparseCM12CurrentMapVelocityLayout {
  const count = map.nodeDimensions.reduce((product, n) => product * n, 1);
  if (!Number.isSafeInteger(count) || count !== map.nodeCount
    || map.nodeDimensions.some(n => !Number.isInteger(n) || n < 1)
    || [map.immutableVelocityBaseWords, map.scratchBaseWords, map.nodalBaseWords]
      .some(base => !Number.isSafeInteger(base) || base < 0 || base + 3 * count > 0xffffffff)) {
    throw new RangeError("Invalid current-map velocity extension arenas");
  }
  const bases = [map.immutableVelocityBaseWords, map.scratchBaseWords, map.nodalBaseWords].sort((a, b) => a-b);
  if (bases[0]! + 3 * count > bases[1]! || bases[1]! + 3 * count > bases[2]!) {
    throw new RangeError("Current-map velocity extension arenas overlap");
  }
  return Object.freeze({ dimensions: map.nodeDimensions, nodeCount: count,
    velocityBaseWords: map.immutableVelocityBaseWords, scratchBaseWords: map.scratchBaseWords,
    fixedMaskBaseWords: map.nodalBaseWords });
}

/** Extends the actual frozen VEX field into its invalid region. Valid samples
 * are Dirichlet data, including physically stationary zero-velocity samples.
 * Unknown nodes undergo a bounded discrete harmonic relaxation. An unknown
 * is never promoted to fixed merely because a sweep gives it a value.
 *
 * The host must initialize every node, execute 32 complete ToScratch /
 * FromScratch pairs, and only then advance map nodes. The map orientation
 * certificate remains authoritative if this bounded relaxation is inadequate.
 */
export function createSparseCM12CurrentMapVelocityWGSL(map: SparseCM12CurrentMapLayout): string {
  const layout = createSparseCM12CurrentMapVelocityLayout(map);
  return /* wgsl */ `
const CM12_CURRENT_MAP_VELOCITY_FIXED_BASE:u32=${layout.fixedMaskBaseWords}u;
fn cm12CurrentMapInitializeVelocity(id:u32,sample:vec4f){
  let fixed=sample.w>0.0;
  // Validity is independent of speed. A resting pool supplies fixed zeros.
  state[CM12_CURRENT_MAP_VELOCITY_FIXED_BASE+3u*id]=select(0.0,1.0,fixed);
  cm12CurrentMapWrite(CM12_CURRENT_MAP_VELOCITY_BASE,id,
    select(vec3f(0.0),sample.xyz,fixed));
}
fn cm12CurrentMapRelaxVelocity(id:u32,source:u32,destination:u32){
  if(state[CM12_CURRENT_MAP_VELOCITY_FIXED_BASE+3u*id]>0.5){
    // Copy without arithmetic so every original VEX value remains exact.
    cm12CurrentMapWrite(destination,id,cm12CurrentMapRead(source,id));return;
  }
  let q=vec3i(cm12CurrentMapNodeCoordinate(id));var sum=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){
    var lo=q;var hi=q;lo[axis]-=1;hi[axis]+=1;
    if(lo[axis]>=0){sum+=cm12CurrentMapRead(source,cm12CurrentMapNodeId(vec3u(lo)));}
    if(hi[axis]<i32(CM12_CURRENT_MAP_NODES[axis])){
      sum+=cm12CurrentMapRead(source,cm12CurrentMapNodeId(vec3u(hi)));
    }
  }
  // Missing neighbours belong to the zero exterior of the padded map grid,
  // not to the physical tank boundary. The existing physical boundary hook
  // supplies impermeability and the smooth exterior velocity taper.
  cm12CurrentMapWrite(destination,id,sum/6.0);
}
@compute @workgroup_size(64)
fn extendCurrentMapVelocityToScratch(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_NODE_COUNT){return;}
  cm12CurrentMapRelaxVelocity(id,CM12_CURRENT_MAP_VELOCITY_BASE,CM12_CURRENT_MAP_SCRATCH_BASE);
}
@compute @workgroup_size(64)
fn extendCurrentMapVelocityFromScratch(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_NODE_COUNT){return;}
  cm12CurrentMapRelaxVelocity(id,CM12_CURRENT_MAP_SCRATCH_BASE,CM12_CURRENT_MAP_VELOCITY_BASE);
}
`;
}
