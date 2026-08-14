export const octreeLosassoVelocityExtensionWGSL = /* wgsl */ `
struct Params {
  faceCapacity: u32,
  width: u32,
  sweep: u32,
  causalFront: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Shared coarse authority: [epoch, rowCount, faceCount, valid, ...]
@group(0) @binding(1) var<storage, read> authority: array<u32>;
// vec4u(axis, flags, bitcast(abs(phi)), layer)
@group(0) @binding(2) var<storage, read> faceMetrics: array<vec4u>;
@group(0) @binding(3) var<storage, read> adjacencyOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> adjacencyFaces: array<u32>;
@group(0) @binding(5) var<storage, read> inputVelocity: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputVelocity: array<f32>;

// A six-neighbor sum is accumulated as signed radix-256 digits. The local
// integer fold is exact and commutative, so rotating the geometric adjacency
// cannot change a last bit merely by permuting x/y/z neighbor order.
const LOCAL_LIMBS:u32=36u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn addLocalF32(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);
 let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;
 let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);
 let shift=select(3u,rawExponent+2u,rawExponent!=0u);let first=shift>>3u;
 let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<LOCAL_LIMBS){(*limbs)[limb]+=sign*byte;}}}
fn localF32(limbsInput:array<i32,36>)->f32{var limbs=limbsInput;
 for(var limb=0u;limb+1u<LOCAL_LIMBS;limb+=1u){let normalized=floorDiv256(limbs[limb]);
  limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
 let negative=limbs[LOCAL_LIMBS-1u]<0;if(negative){for(var limb=0u;limb<LOCAL_LIMBS;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<LOCAL_LIMBS;limb+=1u){let normalized=floorDiv256(limbs[limb]);
   limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<LOCAL_LIMBS;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
 return select(magnitude,-magnitude,negative);}

@compute @workgroup_size(64)
fn jacobiLosassoAxisFaces(@builtin(workgroup_id) group:vec3u,
    @builtin(local_invocation_index) lane:u32) {
  let face = 64u * (group.x + 65535u * group.y) + lane;
  if (authority[3] != 1u || face >= authority[2] || face >= params.faceCapacity) {
    return;
  }
  let metric = faceMetrics[face];
  let flags = metric.y;
  let layer = metric.w;
  // bit 0: projected/interface seed. Solid faces may also be seeds: the value
  // has already been aperture-weighted by the coarse projection.
  if ((flags & 1u) != 0u || layer == 0u || layer > params.width) {
    outputVelocity[face] = inputVelocity[face];
    return;
  }
  // Bit one is the explicit band-membership flag. Seed-only legacy sources
  // remain active because bit zero takes the early path above.
  if ((flags & 2u) == 0u) { outputVelocity[face] = 0.0; return; }
  // Fixed-K constant extension. Input invalidity is explicit, so every sweep
  // advances the finite front by at most one same-axis graph edge. Do not use
  // ceil(|phi|/h) as a causal rank: equal-distance plateaus are ordinary on a
  // Cartesian lattice and a strict lower-rank predicate disconnects diagonal
  // W7 support even when the graph itself is complete.
  var exact:array<i32,36>;
  var count = 0u;
  let begin = adjacencyOffsets[face];
  let end = adjacencyOffsets[face + 1u];
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let neighbor = adjacencyFaces[cursor];
    if (neighbor == 0xffffffffu || neighbor >= authority[2]) { continue; }
    let neighborMetric = faceMetrics[neighbor];
    if (neighborMetric.x != metric.x || (neighborMetric.y & 3u) == 0u) {
      continue;
    }
    // A starved predecessor is not evidence for this layer. Other causal
    // paths may still reach the face, so exclude the sentinel from both the
    // exact sum and its divisor instead of propagating non-finite storage.
    let neighborVelocity = inputVelocity[neighbor];
    if (!finite(neighborVelocity)) { continue; }
    addLocalF32(&exact,neighborVelocity);
    count += 1u;
  }
  // Preserve the distinction between a physical zero and an unreachable band
  // face. The sentinel is carried by later causal sweeps and consumed only as
  // a request for coarse-face reconstruction by the staging pass.
  if (count > 0u) {
    outputVelocity[face] = localF32(exact) / f32(count);
  } else {
    // WGSL rejects a NaN-valued module constant, so form the storage sentinel
    // in this runtime branch. A non-finite value means "no extension path
    // reached this face"; staging reconstructs it from the coarse bracket.
    var starvedBits = 0x7fc00000u;
    outputVelocity[face] = bitcast<f32>(starvedBits);
  }
}
`;
