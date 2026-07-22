/**
 * Solver-grid cross-section rendered as an independent presentation layer.
 *
 * Keeping this out of both water renderers lets the same scientific overlay
 * compose over raster optics. The shared view
 * uniform supplies the slice axis/position through `debug.xy`; this pipeline
 * only owns the grid-specific sampling and alpha blend.
 */

import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import type { SparseSurfaceBandGPUSource } from "./webgpu-sparse-surface-band";
import { OCTREE_CONSUMER_MAX_FACE_CANDIDATES, validateUnifiedOctreeConsumerSource, type UnifiedOctreeConsumerSource } from "./octree-consumer-sampling";
import { OCTREE_SURFACE_LEAF_RECORD_BYTES } from "./webgpu-octree-surface-pages";

export const gridOverlayShader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
  gridInfo: vec4f,
  debug: vec4f,
  // environment.x is the art-direction preset (read by the water shaders);
  // .y carries the solver's last substep dt and .z its max liquid speed so
  // the field modes can color CFL and normalized speed without any readback.
  // .w identifies the quadtree optical strategy: 0 = unavailable,
  // 1 = fixed quarter-depth, 2 = motion-adaptive.
  environment: vec4f,
}
struct BodyGPU {
  positionRadius: vec4f,
  halfSizeShape: vec4f,
  orientation: vec4f,
  colorSelected: vec4f,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> bodies: array<BodyGPU, 12>;
@group(0) @binding(2) var fluidField: texture_3d<f32>;
@group(0) @binding(3) var tallCellBases: texture_2d<f32>;
@group(0) @binding(4) var adaptiveCells: texture_3d<u32>;
@group(0) @binding(5) var velocityField: texture_3d<f32>;
@group(0) @binding(6) var pressureSamples: texture_3d<u32>;
@group(0) @binding(7) var divergenceField: texture_3d<f32>;
@group(0) @binding(8) var mappedPressureField: texture_3d<f32>;
struct SparseSurfaceParams {
  coarseDims: vec4u,
  fineDims: vec4u,
  brickDims: vec4u,
  settings: vec4f,
  cellAndDt: vec4f,
  sizing: vec4f,
  physical: vec4f,
}
@group(0) @binding(9) var<storage, read> sparseSurfacePages: array<u32>;
@group(0) @binding(10) var<uniform> sparseSurfaceParams: SparseSurfaceParams;
@group(0) @binding(11) var<storage, read> sparseSurfacePhi: array<f32>;
@group(0) @binding(12) var<storage, read> sparseSurfaceControl: array<u32>;
@group(0) @binding(13) var<storage, read> sparseSurfaceStates: array<u32>;
struct OctreeSurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct OctreeSurfaceParams { shape:vec4u, offsets0:vec4u, offsets1:vec4u, offsets2:vec4u, cellDt:vec4f, spare0:vec4u, spare1:vec4u, spare2:vec4u }
struct OctreeFaceRecord { negativeRow:u32,positiveRow:u32,originX:u32,originY:u32,originZ:u32,axisSpan:u32,normalVelocity:f32,area:f32 }
@group(0) @binding(14) var<storage, read> octreeSurfaceLeaves:array<OctreeSurfaceLeaf>;
@group(0) @binding(15) var<storage, read> octreeSurfaceArena:array<u32>;
@group(0) @binding(16) var<uniform> octreeSurfaceParams:OctreeSurfaceParams;
@group(0) @binding(17) var<storage, read> octreeFaces:array<OctreeFaceRecord>;
@group(0) @binding(18) var<storage, read> octreeIncidence:array<u32>;
@group(0) @binding(19) var<storage, read> octreeFaceControl:array<u32>;

const SPARSE_SURFACE_INVALID: u32 = 0xffffffffu;
const OCTREE_FACE_INCIDENCE_PER_ROW: u32 = ${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}u;

fn octreeSurfaceBound()->bool{let r=octreeSurfaceParams.shape.z;return (r==2u||r==4u)&&arrayLength(&octreeSurfaceArena)>6u&&octreeSurfaceParams.shape.x<=arrayLength(&octreeSurfaceLeaves);}
fn octreeSurfaceAvailable()->bool{return octreeSurfaceBound()&&octreeSurfaceArena[3]==0u;}
fn octreeOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn octreeLeafOrigin(leaf:OctreeSurfaceLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn octreeHash(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn octreeLeafContains(row:u32,point:vec3f)->bool{if(row>=octreeSurfaceParams.shape.x){return false;}let leaf=octreeSurfaceLeaves[row];let origin=vec3f(octreeLeafOrigin(leaf));return leaf.size>0u&&all(point>=origin)&&all(point<origin+vec3f(f32(leaf.size)));}
fn octreeSurfaceRow(cell:vec3i)->u32{
  if(!octreeSurfaceBound()||any(cell<vec3i(0))){return SPARSE_SURFACE_INVALID;}
  let point=vec3f(cell)+vec3f(0.5);let hinted=textureLoad(pressureSamples,cell,0).x;
  // Reuse the owner row already produced by the overlay materialization pass.
  // Only air-side halo cells pay for a bounded compact-hash lookup.
  if(octreeLeafContains(hinted,point)){return hinted;}
  let mask=octreeSurfaceParams.offsets2.y-1u;
  for(var size=1u;size<=1024u;size<<=1u){let q=vec3u(floor(point/f32(size)));var slot=octreeHash(q)&mask;
    for(var probe=0u;probe<32u;probe+=1u){let encoded=octreeSurfaceArena[octreeSurfaceParams.offsets1.y+slot];if(encoded==0u){break;}let row=encoded-1u;if(octreeLeafContains(row,point)){return row;}slot=(slot+1u)&mask;}
  }
  return SPARSE_SURFACE_INVALID;
}
fn octreeLeafFallback(row:u32,point:vec3f)->f32{let leaf=octreeSurfaceLeaves[row];let origin=vec3f(octreeLeafOrigin(leaf));let centre=origin+vec3f(0.5*f32(leaf.size));let physicalGradient=leaf.phiGradient.yzw/max(octreeSurfaceParams.cellDt.xyz,vec3f(1e-9));let boundedPhysical=physicalGradient/max(1.0,length(physicalGradient));return leaf.phiGradient.x+dot(boundedPhysical*octreeSurfaceParams.cellDt.xyz,point-centre);}
fn octreeSurfaceLoad(base:u32,q:vec3u)->f32{let r=octreeSurfaceParams.shape.z;return bitcast<f32>(octreeSurfaceArena[base+q.x+r*(q.y+r*q.z)]);}
fn octreeSurfacePhi(row:u32,point:vec3f)->f32{
  if(row==SPARSE_SURFACE_INVALID||row>=octreeSurfaceParams.shape.x){return 1e20;}let slot=octreeSurfaceArena[octreeSurfaceParams.offsets0.x+row];if(slot==SPARSE_SURFACE_INVALID||slot>=octreeSurfaceParams.shape.y){return octreeLeafFallback(row,point);}
  let leaf=octreeSurfaceLeaves[row];let r=octreeSurfaceParams.shape.z;let origin=vec3f(octreeLeafOrigin(leaf));let grid=clamp((point-origin)/f32(leaf.size)*f32(r)-vec3f(0.5),vec3f(0.0),vec3f(f32(r-1u)));let a=vec3u(floor(grid));let b=min(a+vec3u(1u),vec3u(r-1u));let t=fract(grid);let base=octreeSurfaceParams.offsets1.z+slot*octreeSurfaceParams.shape.w;
  return mix(mix(mix(octreeSurfaceLoad(base,a),octreeSurfaceLoad(base,vec3u(b.x,a.y,a.z)),t.x),mix(octreeSurfaceLoad(base,vec3u(a.x,b.y,a.z)),octreeSurfaceLoad(base,vec3u(b.x,b.y,a.z)),t.x),t.y),mix(mix(octreeSurfaceLoad(base,vec3u(a.x,a.y,b.z)),octreeSurfaceLoad(base,vec3u(b.x,a.y,b.z)),t.x),mix(octreeSurfaceLoad(base,vec3u(a.x,b.y,b.z)),octreeSurfaceLoad(base,b),t.x),t.y),t.z);
}
fn octreeSurfaceCoreSample(cell:vec3i)->bool{
  let row=octreeSurfaceRow(cell);if(row==SPARSE_SURFACE_INVALID){return false;}let state=octreeSurfaceArena[octreeSurfaceParams.offsets0.y+row];
  return (state&2u)!=0u&&abs(octreeSurfacePhi(row,vec3f(cell)+vec3f(0.5)))<=1.5*min(octreeSurfaceParams.cellDt.x,min(octreeSurfaceParams.cellDt.y,octreeSurfaceParams.cellDt.z));
}

// Overflow means the bounded fine pool could not cover the complete surface.
// The renderer falls back to the complete coarse field in that case, and the
// grid view follows the same rule instead of displaying a misleading partial
// fine hierarchy.
fn sparseSurfaceAvailable() -> bool {
  return sparseSurfaceParams.coarseDims.w > 0u
    && arrayLength(&sparseSurfaceControl) > 2u
    && sparseSurfaceControl[2] == 0u;
}
fn sparseSurfacePayload(fineCell: vec3i) -> u32 {
  if (!sparseSurfaceAvailable() || any(fineCell < vec3i(0)) || any(fineCell >= vec3i(sparseSurfaceParams.fineDims.xyz))) { return SPARSE_SURFACE_INVALID; }
  let q=vec3u(fineCell);let brickSize=sparseSurfaceParams.fineDims.w;let page=q/brickSize;
  let index=page.x+sparseSurfaceParams.brickDims.x*(page.y+sparseSurfaceParams.brickDims.y*page.z);
  if (index >= arrayLength(&sparseSurfacePages)) { return SPARSE_SURFACE_INVALID; }
  let slot = sparseSurfacePages[index];
  if (slot == SPARSE_SURFACE_INVALID || slot >= u32(sparseSurfaceParams.sizing.w)) { return SPARSE_SURFACE_INVALID; }
  let local=q%brickSize;let localIndex=local.x+brickSize*(local.y+brickSize*local.z);
  let payload=slot*brickSize*brickSize*brickSize+localIndex;
  return select(SPARSE_SURFACE_INVALID,payload,payload<arrayLength(&sparseSurfacePhi));
}
fn sparseSurfacePageState(fineCell: vec3i) -> u32 {
  if (!sparseSurfaceAvailable() || any(fineCell < vec3i(0)) || any(fineCell >= vec3i(sparseSurfaceParams.fineDims.xyz))) { return 0u; }
  let q = vec3u(fineCell); let page = q / sparseSurfaceParams.fineDims.w;
  let pageIndex = page.x + sparseSurfaceParams.brickDims.x * (page.y + sparseSurfaceParams.brickDims.y * page.z);
  if (pageIndex >= arrayLength(&sparseSurfaceStates)) { return 0u; }
  return sparseSurfaceStates[pageIndex];
}
fn sparseSurfaceCoreSample(fineCell: vec3i) -> bool {
  if ((sparseSurfacePageState(fineCell) & 2u) == 0u) { return false; }
  let payload=sparseSurfacePayload(fineCell);
  if (payload == SPARSE_SURFACE_INVALID) { return false; }
  let factor=f32(sparseSurfaceParams.coarseDims.w);
  let fineH=min(sparseSurfaceParams.cellAndDt.x,min(sparseSurfaceParams.cellAndDt.y,sparseSurfaceParams.cellAndDt.z))/factor;
  // Residency includes whole 8^3 support bricks and a stencil halo.  Pink is
  // intentionally the much smaller physical shell around phi=0, not every
  // allocated support voxel.
  return abs(sparseSurfacePhi[payload]) <= 1.5*fineH;
}

struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var result: VertexOutput;
  result.position = vec4f(positions[index], 0.0, 1.0);
  result.uv = positions[index] * 0.5 + 0.5;
  return result;
}

fn boxIntersection(ro: vec3f, rd: vec3f, boundsMin: vec3f, boundsMax: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let a = (boundsMin - ro) * inv;
  let b = (boundsMax - ro) * inv;
  let near3 = min(a, b);
  let far3 = max(a, b);
  return vec2f(max(max(near3.x, near3.y), near3.z), min(min(far3.x, far3.y), far3.z));
}
fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.yzw, v);
  return v + q.x * t + cross(q.yzw, t);
}
fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f { return quatRotate(vec4f(q.x, -q.yzw), v); }

fn bodyDistance(ro: vec3f, rd: vec3f, body: BodyGPU) -> f32 {
  let origin = quatInverseRotate(body.orientation, ro - body.positionRadius.xyz);
  let direction = quatInverseRotate(body.orientation, rd);
  let shape = i32(round(body.halfSizeShape.w));
  if (shape == 0) {
    let radius = body.halfSizeShape.x;
    let projected = dot(origin, direction);
    let discriminant = projected * projected - dot(origin, origin) + radius * radius;
    if (discriminant < 0.0) { return 1e20; }
    let root = sqrt(discriminant);
    let near = -projected - root;
    let far = -projected + root;
    let candidate = select(far, near, near > 1e-4);
    return select(1e20, candidate, candidate > 1e-4);
  }
  if (shape == 1) {
    let hit = boxIntersection(origin, direction, -body.halfSizeShape.xyz, body.halfSizeShape.xyz);
    if (hit.x > hit.y) { return 1e20; }
    let candidate = select(hit.y, hit.x, hit.x > 1e-4);
    return select(1e20, candidate, candidate > 1e-4);
  }
  let radius = body.halfSizeShape.x;
  let halfHeight = body.halfSizeShape.y;
  var best = 1e20;
  let a = dot(direction.xz, direction.xz);
  let b = dot(origin.xz, direction.xz);
  let c = dot(origin.xz, origin.xz) - radius * radius;
  if (a > 1e-7) {
    let discriminant = b * b - a * c;
    if (discriminant >= 0.0) {
      let side = (-b - sqrt(discriminant)) / a;
      let y = origin.y + direction.y * side;
      if (side > 1e-4 && abs(y) <= halfHeight) { best = side; }
    }
  }
  if (shape == 2) {
    for (var end = -1.0; end <= 1.0; end += 2.0) {
      let offset = origin - vec3f(0.0, end * halfHeight, 0.0);
      let projected = dot(offset, direction);
      let discriminant = projected * projected - dot(offset, offset) + radius * radius;
      if (discriminant >= 0.0) {
        let cap = -projected - sqrt(discriminant);
        if (cap > 1e-4 && cap < best) { best = cap; }
      }
    }
  } else if (abs(direction.y) > 1e-7) {
    for (var end = -1.0; end <= 1.0; end += 2.0) {
      let cap = (end * halfHeight - origin.y) / direction.y;
      let point = origin + direction * cap;
      if (cap > 1e-4 && cap < best && dot(point.xz, point.xz) <= radius * radius) { best = cap; }
    }
  }
  return best;
}

fn nearestBodyDistance(ro: vec3f, rd: vec3f) -> f32 {
  var nearest = 1e20;
  let bodyCount = u32(round(u.options.z));
  for (var index = 0u; index < 12u; index += 1u) {
    if (index >= bodyCount) { break; }
    nearest = min(nearest, bodyDistance(ro, rd, bodies[index]));
  }
  return nearest;
}

fn fluidSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(cell, vec3i(0), dims - vec3i(1));
  if(octreeSurfaceAvailable()){let phi=octreeSurfacePhi(octreeSurfaceRow(q),vec3f(q)+vec3f(0.5));if(phi>=1e19){return select(0.0,1.0,hasLiquidPressureDof(q));}let h=u.container.y/max(u.gridInfo.y,1.0);return clamp(0.5-phi/max(h,1e-9),0.0,1.0);}
  if (u.gridInfo.w < 1.5) { return textureLoad(fluidField, q, 0).x; }
  let cellSizeY = u.container.y / max(u.gridInfo.y, 1.0);
  if (u.gridInfo.w > 2.5) { return clamp(0.5 - textureLoad(fluidField, q, 0).x / cellSizeY, 0.0, 1.0); }
  let base = i32(round(textureLoad(tallCellBases, q.xz, 0).x));
  if (q.y < base && base > 0) { let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);let phi=mix(textureLoad(fluidField,vec3i(q.x,0,q.z),0).x,textureLoad(fluidField,vec3i(q.x,1,q.z),0).x,t);return clamp(0.5-phi/cellSizeY,0.0,1.0); }
  let packedY = 2 + q.y - base;
  let stored = vec3i(textureDimensions(fluidField));
  if (packedY < 2 || packedY >= stored.y) { return 0.0; }
  return clamp(0.5-textureLoad(fluidField, vec3i(q.x, packedY, q.z), 0).x/cellSizeY,0.0,1.0);
}

fn levelSetSample(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz); let q = clamp(cell, vec3i(0), dims - vec3i(1));
  if(octreeSurfaceAvailable()){return octreeSurfacePhi(octreeSurfaceRow(q),vec3f(q)+vec3f(0.5));}
  let h = u.container.y / max(u.gridInfo.y, 1.0);
  if (u.gridInfo.w > 2.5) { return textureLoad(fluidField, q, 0).x; }
  return (0.5 - fluidSample(q)) * 4.0 * h;
}

fn hasLiquidPressureDof(cell: vec3i) -> bool {
  let samples = textureLoad(pressureSamples, cell, 0);
  return samples.x != 0xffffffffu || samples.y != 0xffffffffu;
}

fn adaptiveCellKey(cell: vec3i, dims: vec3i) -> vec2u {
  return textureLoad(adaptiveCells, clamp(cell, vec3i(0), dims - vec3i(1)), 0).xy;
}

// Returns [lower y, upper y, horizontal quadtree leaf size]. The segmentation
// encoder emits every retained optical row as its own one-cell-high segment;
// only uncut runs are vertically merged. Reading this from the ownership
// texture therefore visualizes the conservative layer actually consumed by
// the pressure solve, including changes introduced by quadtree reduction.
fn adaptiveCellVerticalShape(cell: vec3i, dims: vec3i) -> vec3i {
  let key = adaptiveCellKey(cell, dims);
  return vec3i(i32(key.y & 1023u), i32((key.y >> 10u) & 1023u), max(1, i32((key.x >> 20u) & 1023u)));
}

fn isOpticalCube(cell: vec3i, dims: vec3i) -> bool {
  let shape = adaptiveCellVerticalShape(cell, dims);
  return shape.y - shape.x == 1;
}

// World-cell velocity through the same packed mapping as fluidSample. The
// tall-cell interior uses the solver's piecewise reconstruction (top world
// cell = top endpoint dof, the rest = bottom dof) so the displayed field is
// the one the projection actually controls.
fn velocitySample(cell: vec3i) -> vec3f {
  let dims = vec3i(u.gridInfo.xyz);
  let q = clamp(cell, vec3i(0), dims - vec3i(1));
  if (u.gridInfo.w < 1.5 || u.gridInfo.w > 2.5) { return textureLoad(velocityField, q, 0).xyz; }
  let base = i32(round(textureLoad(tallCellBases, q.xz, 0).x));
  if (q.y < base && base > 0) {
    let row = select(0, 1, q.y == base - 1);
    return textureLoad(velocityField, vec3i(q.x, row, q.z), 0).xyz;
  }
  let packedY = 2 + q.y - base;
  let stored = vec3i(textureDimensions(velocityField));
  if (packedY < 2 || packedY >= stored.y) { return vec3f(0.0); }
  return textureLoad(velocityField, vec3i(q.x, packedY, q.z), 0).xyz;
}

struct OctreeFaceAudit { velocity:vec3f, count:u32, axisMask:u32, fault:u32 }
fn octreeFaceAudit(row:u32,point:vec3f)->OctreeFaceAudit{
  var result=OctreeFaceAudit(vec3f(0.0),0u,0u,0u);
  if(arrayLength(&octreeFaceControl)<6u||octreeFaceControl[1]!=0u||octreeFaceControl[0]>octreeFaceControl[2]){result.fault|=1u;return result;}
  let rowCapacity=octreeFaceControl[3];if(row>=rowCapacity||row>=arrayLength(&octreeIncidence)){result.fault|=2u;return result;}
  let publishedCount=octreeIncidence[row];let count=min(publishedCount,OCTREE_FACE_INCIDENCE_PER_ROW);if(publishedCount>OCTREE_FACE_INCIDENCE_PER_ROW){result.fault|=4u;}
  var weighted=vec3f(0.0);var weights=vec3f(0.0);
  for(var local=0u;local<count;local+=1u){let at=rowCapacity+row*OCTREE_FACE_INCIDENCE_PER_ROW+local;if(at>=arrayLength(&octreeIncidence)){result.fault|=8u;break;}let faceIndex=octreeIncidence[at];if(faceIndex>=octreeFaceControl[0]||faceIndex>=octreeFaceControl[2]||faceIndex>=arrayLength(&octreeFaces)){result.fault|=8u;continue;}
    let face=octreeFaces[faceIndex];let axis=face.axisSpan&3u;let span=face.axisSpan>>2u;if(axis>2u||span==0u||abs(face.normalVelocity)>3.402823e38||!(face.negativeRow==row||face.positiveRow==row)){result.fault|=8u;continue;}
    var centre=vec3f(vec3u(face.originX,face.originY,face.originZ));centre[(axis+1u)%3u]+=0.5*f32(span);centre[(axis+2u)%3u]+=0.5*f32(span);let delta=point-centre;let support=f32(max(1u,span));let tent=max(vec3f(0.0),vec3f(1.0)-abs(delta)/support);let weight=tent.x*tent.y*tent.z;
    weighted[axis]+=weight*face.normalVelocity;weights[axis]+=weight;result.axisMask|=1u<<axis;result.count+=1u;
  }
  for(var axis=0u;axis<3u;axis+=1u){if(weights[axis]>0.0){result.velocity[axis]=weighted[axis]/weights[axis];}}
  return result;
}

// Shared five-stop heat ramp for the field modes: deep blue through cyan,
// green, and amber to red as t goes 0..1.
fn heatColor(t: f32) -> vec3f {
  let clamped = clamp(t, 0.0, 1.0);
  if (clamped < 0.25) { return mix(vec3f(0.13, 0.22, 0.55), vec3f(0.06, 0.62, 0.80), clamped * 4.0); }
  if (clamped < 0.5) { return mix(vec3f(0.06, 0.62, 0.80), vec3f(0.22, 0.75, 0.34), (clamped - 0.25) * 4.0); }
  if (clamped < 0.75) { return mix(vec3f(0.22, 0.75, 0.34), vec3f(0.98, 0.82, 0.20), (clamped - 0.5) * 4.0); }
  return mix(vec3f(0.98, 0.82, 0.20), vec3f(0.90, 0.22, 0.15), (clamped - 0.75) * 4.0);
}

struct RepresentedCell {
  lower: vec3f,
  upper: vec3f,
}

fn representedCell(cell: vec3i, dims: vec3i, boundsMin: vec3f, size: vec3f, adaptiveGrid: bool, tallGrid: bool) -> RepresentedCell {
  var lower = cell;
  var upper = cell + vec3i(1);
  if (adaptiveGrid) {
    let key = adaptiveCellKey(cell, dims);
    lower = vec3i(i32(key.x & 1023u), i32(key.y & 1023u), i32((key.x >> 10u) & 1023u));
    let leafSize = i32((key.x >> 20u) & 1023u);
    upper = vec3i(lower.x + leafSize, i32((key.y >> 10u) & 1023u), lower.z + leafSize);
  } else if (tallGrid) {
    let base = i32(round(textureLoad(tallCellBases, cell.xz, 0).x));
    if (cell.y < base) {
      lower.y = 0;
      upper.y = base;
    }
  }
  let cellSize = size / vec3f(dims);
  return RepresentedCell(boundsMin + vec3f(lower) * cellSize, boundsMin + vec3f(upper) * cellSize);
}

fn bodySignedDistance(point: vec3f, body: BodyGPU) -> f32 {
  let local = quatInverseRotate(body.orientation, point - body.positionRadius.xyz);
  let shape = i32(round(body.halfSizeShape.w));
  if (shape == 0) { return length(local) - body.halfSizeShape.x; }
  if (shape == 1) {
    let q = abs(local) - body.halfSizeShape.xyz;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  }
  if (shape == 2) {
    let axisPoint = vec3f(0.0, clamp(local.y, -body.halfSizeShape.y, body.halfSizeShape.y), 0.0);
    return length(local - axisPoint) - body.halfSizeShape.x;
  }
  let q = vec2f(length(local.xz) - body.halfSizeShape.x, abs(local.y) - body.halfSizeShape.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

struct GridBodySample {
  occupied: bool,
  selected: f32,
}

fn gridBodySample(cellBounds: RepresentedCell) -> GridBodySample {
  let center = (cellBounds.lower + cellBounds.upper) * 0.5;
  let cellRadius = length((cellBounds.upper - cellBounds.lower) * 0.5);
  var nearest = 1e20;
  var result = GridBodySample(false, 0.0);
  let bodyCount = u32(round(u.options.z));
  for (var index = 0u; index < 12u; index += 1u) {
    if (index >= bodyCount) { break; }
    // A represented cell is solid when its circumscribed volume reaches the
    // analytic primitive. This conservative voxelization keeps small bodies
    // visible and, for adaptive grids, fills the complete quadtree leaf rather
    // than leaking the hidden dense backing-grid resolution into the view.
    let closest = clamp(bodies[index].positionRadius.xyz, cellBounds.lower, cellBounds.upper);
    let sphereDistance = length(closest - bodies[index].positionRadius.xyz) - bodies[index].positionRadius.w;
    let shapeDistance = bodySignedDistance(center, bodies[index]) - cellRadius;
    let distance = max(sphereDistance, shapeDistance);
    if (distance <= cellRadius && distance < nearest) {
      nearest = distance;
      result = GridBodySample(true, bodies[index].colorSelected.w);
    }
  }
  return result;
}

struct GridSample {
  color: vec3f,
  alpha: f32,
  solid: bool,
}

fn gridSample(point: vec3f, boundsMin: vec3f, size: vec3f, axis: i32, footprint: f32) -> GridSample {
  let dims = vec3i(u.gridInfo.xyz);
  let local3 = clamp((point - boundsMin) / size, vec3f(0.0), vec3f(0.99999)) * vec3f(dims);
  let cell = clamp(vec3i(floor(local3)), vec3i(0), dims - vec3i(1));
  var samplePosition = local3.xy;
  var cellPerPixel = vec2f(footprint * f32(dims.x) / size.x, footprint * f32(dims.y) / size.y);
  var firstPlaneAxis = 0;
  var secondPlaneAxis = 1;
  if (axis == 2) {
    samplePosition.x = local3.z;
    cellPerPixel.x = footprint * f32(dims.z) / size.z;
    firstPlaneAxis = 2;
  } else if (axis == 3) {
    samplePosition = local3.xz;
    cellPerPixel = vec2f(footprint * f32(dims.x) / size.x, footprint * f32(dims.z) / size.z);
    secondPlaneAxis = 2;
  }
  let derivative = max(cellPerPixel, vec2f(1e-5));
  let pixelsPerCell = 1.0 / max(derivative.x, derivative.y);
  let lineFade = smoothstep(2.5, 6.0, pixelsPerCell);
  let dotFade = smoothstep(9.0, 18.0, pixelsPerCell);
  let adaptiveGrid = u.debug.z > 0.5;
  let tallGrid = u.gridInfo.w > 1.5 && u.gridInfo.w < 2.5;
  let octreeGrid = u.gridInfo.w > 2.5;
  var base = 0.0;
  if (tallGrid) { base = round(textureLoad(tallCellBases, cell.xz, 0).x); }
  let stored = vec3i(textureDimensions(fluidField));
  let bandLayers = select(dims.y, stored.y - 2, tallGrid);
  let bandTop = min(i32(base) + bandLayers, dims.y);
  let firstGridLine = 1.0 - smoothstep(0.4, 1.2, (0.5 - abs(fract(samplePosition.x) - 0.5)) / derivative.x);
  let secondGridLine = 1.0 - smoothstep(0.4, 1.2, (0.5 - abs(fract(samplePosition.y) - 0.5)) / derivative.y);
  var fill = vec3f(0.0);
  var alpha = 0.0;
  var line = 0.0;
  var sampleDot = 0.0;
  var opticalBoundary = 0.0;
  if (adaptiveGrid) {
    // The simulation transports its fields on a dense cubic backing texture,
    // but pressure is represented by adaptive quadtree/tall cells. The id
    // texture maps each backing voxel to that represented cell. Only an id
    // transition is a real grid edge; the hidden fine-grid boundaries must
    // not appear in this scientific overlay.
    let own = adaptiveCellKey(cell, dims);
    var lowerFirst = cell;
    var upperFirst = cell;
    lowerFirst[firstPlaneAxis] -= 1;
    upperFirst[firstPlaneAxis] += 1;
    var lowerFirstEdge = cell[firstPlaneAxis] == 0;
    var upperFirstEdge = cell[firstPlaneAxis] == dims[firstPlaneAxis] - 1;
    if (!lowerFirstEdge) { lowerFirstEdge = any(adaptiveCellKey(lowerFirst, dims) != own); }
    if (!upperFirstEdge) { upperFirstEdge = any(adaptiveCellKey(upperFirst, dims) != own); }
    var lowerSecond = cell;
    var upperSecond = cell;
    lowerSecond[secondPlaneAxis] -= 1;
    upperSecond[secondPlaneAxis] += 1;
    var lowerSecondEdge = cell[secondPlaneAxis] == 0;
    var upperSecondEdge = cell[secondPlaneAxis] == dims[secondPlaneAxis] - 1;
    if (!lowerSecondEdge) { lowerSecondEdge = any(adaptiveCellKey(lowerSecond, dims) != own); }
    if (!upperSecondEdge) { upperSecondEdge = any(adaptiveCellKey(upperSecond, dims) != own); }
    let firstFraction = fract(samplePosition.x);
    let secondFraction = fract(samplePosition.y);
    let firstDistance = min(select(1e6, firstFraction / derivative.x, lowerFirstEdge), select(1e6, (1.0 - firstFraction) / derivative.x, upperFirstEdge));
    let secondDistance = min(select(1e6, secondFraction / derivative.y, lowerSecondEdge), select(1e6, (1.0 - secondFraction) / derivative.y, upperSecondEdge));
    line = 1.0 - smoothstep(0.4, 1.2, min(firstDistance, secondDistance));
    var below = cell;
    var above = cell;
    below.y -= 1;
    above.y += 1;
    var lowerYEdge = cell.y == 0;
    var upperYEdge = cell.y == dims.y - 1;
    if (!lowerYEdge) { lowerYEdge = any(adaptiveCellKey(below, dims) != own); }
    if (!upperYEdge) { upperYEdge = any(adaptiveCellKey(above, dims) != own); }
    let isTall = !lowerYEdge || !upperYEdge;
    let wet = fluidSample(cell) > 0.5;
    fill = select(select(vec3f(0.85, 0.91, 0.89), vec3f(0.20, 0.50, 0.74), wet), select(vec3f(0.10, 0.23, 0.22), vec3f(0.03, 0.52, 0.47), wet), isTall);
    // The octree deliberately refines several dry cells around phi=0. Filling
    // those cells makes the valid air-side refinement band read as a milky
    // slab. Keep all octree air outline-only while preserving filled liquid
    // leaves and the existing quadtree/tall-cell presentation.
    let dryAlpha = select(select(0.08, 0.03, isTall), 0.0, octreeGrid);
    let wetAlpha = select(0.55, 0.78, isTall);
    alpha = select(dryAlpha, wetAlpha, wet);
  } else if (axis == 3) {
    let wet = fluidSample(cell) > 0.5;
    if (cell.y < i32(base)) {
      fill = select(vec3f(0.10, 0.23, 0.22), vec3f(0.03, 0.52, 0.47), wet);
      alpha = select(0.40, 0.78, wet);
    } else if (cell.y < bandTop) {
      fill = select(vec3f(0.85, 0.91, 0.89), vec3f(0.20, 0.50, 0.74), wet);
      alpha = select(0.18, 0.55, wet);
      let distance = length(fract(samplePosition) - vec2f(0.5));
      sampleDot = (1.0 - smoothstep(0.17, 0.17 + max(derivative.x, derivative.y) * 1.6, distance)) * dotFade;
    } else {
      let stripe = smoothstep(0.38, 0.5, abs(fract((samplePosition.x + samplePosition.y) * 0.25) - 0.5));
      fill = vec3f(0.62, 0.24, 0.22);
      alpha = 0.08 + 0.10 * stripe;
    }
    line = max(firstGridLine, secondGridLine);
  } else if (cell.y < i32(base)) {
    let wet = fluidSample(cell) > 0.5;
    fill = select(vec3f(0.10, 0.23, 0.22), vec3f(0.03, 0.52, 0.47), wet);
    alpha = select(0.40, 0.78, wet);
    let baseEdge = 1.0 - smoothstep(0.4, 1.4, min(samplePosition.y, abs(base - samplePosition.y)) / derivative.y);
    line = max(firstGridLine * lineFade * 0.45, baseEdge);
    let dy = min(abs(samplePosition.y - 0.5), abs(samplePosition.y - (base - 0.5)));
    let distance = length(vec2f(fract(samplePosition.x) - 0.5, dy));
    sampleDot = (1.0 - smoothstep(0.17, 0.17 + max(derivative.x, derivative.y) * 1.6, distance)) * dotFade;
  } else if (cell.y < bandTop) {
    let wet = fluidSample(cell) > 0.5;
    fill = select(vec3f(0.85, 0.91, 0.89), vec3f(0.20, 0.50, 0.74), wet);
    alpha = select(0.18, 0.55, wet);
    line = max(firstGridLine, secondGridLine);
    let distance = length(fract(samplePosition.xy) - vec2f(0.5));
    sampleDot = (1.0 - smoothstep(0.17, 0.17 + max(derivative.x, derivative.y) * 1.6, distance)) * dotFade;
  } else {
    let stripe = smoothstep(0.38, 0.5, abs(fract((samplePosition.x + samplePosition.y) * 0.25) - 0.5));
    fill = vec3f(0.62, 0.24, 0.22);
    alpha = 0.08 + 0.10 * stripe;
    line = firstGridLine * 0.35;
  }
  // Field modes recolor represented cells from the live velocity texture;
  // structural lines, sample dots, the above-band hatch, and rigid-body
  // occupancy all stay so the heatmap keeps its spatial reference frame.
  let fieldMode = i32(round(u.debug.w));
  if (fieldMode > 0 && (adaptiveGrid || cell.y < bandTop)) {
    var velocity = velocitySample(cell);
    var compactVelocityFault = false;
    if (octreeGrid && (fieldMode == 1 || fieldMode == 2)) {
      let row = octreeSurfaceRow(cell);let audit = octreeFaceAudit(row, vec3f(cell) + vec3f(0.5));
      velocity = audit.velocity;compactVelocityFault = row == SPARSE_SURFACE_INVALID || audit.fault != 0u || audit.axisMask != 7u;
    }
    let wet = fluidSample(cell) > 0.5;
    if (fieldMode == 1) {
      // Per-cell component CFL at the solver's substep dt: the quantity whose
      // global maximum picks the substep count (substeps = ceil(maxCfl / 2)).
      let h = size / vec3f(dims);
      let dt = max(u.environment.y, 1e-6);
      let cfl = max(abs(velocity.x) * dt / h.x, max(abs(velocity.y) * dt / h.y, abs(velocity.z) * dt / h.z));
      // log2 ramp: cfl 1 -> 0.32, 2 -> 0.5, 4 -> 0.73, 8+ -> 1.
      fill = heatColor(log2(1.0 + cfl) / log2(9.0));
      alpha = select(select(0.30, 0.72, wet), 0.92, cfl > 1.0);
      if (compactVelocityFault && wet) { fill=vec3f(1.0,0.01,0.01);alpha=0.98; }
    } else if (fieldMode == 2) {
      let speed = length(velocity);
      fill = heatColor(speed / max(u.environment.z, 1e-4));
      alpha = select(0.30, 0.85, wet);
      if (compactVelocityFault && wet) { fill=vec3f(1.0,0.01,0.01);alpha=0.98; }
    } else if (fieldMode == 3) {
      let phi = levelSetSample(cell); let h = min(size.x / f32(dims.x), min(size.y / f32(dims.y), size.z / f32(dims.z)));
      let resident=phi<1e19;let signed=clamp(select(4.0*h,phi,resident)/max(4.0*h,1e-6),-1.0,1.0);
      fill = select(mix(vec3f(0.96, 0.96, 0.90), vec3f(0.93, 0.47, 0.16), signed), mix(vec3f(0.96, 0.96, 0.90), vec3f(0.10, 0.45, 0.92), -signed), signed < 0.0);
      alpha = select(0.10,0.80,resident);line=max(line,select(0.0,1.0-smoothstep(0.04*h,0.22*h,abs(phi)),resident));
    } else if (fieldMode == 4) {
      let divergence = textureLoad(divergenceField, cell, 0).x; let scaled = clamp(divergence * max(u.environment.y, 1e-6), -1.0, 1.0);
      fill = select(mix(vec3f(0.96), vec3f(0.88, 0.10, 0.08), scaled), mix(vec3f(0.96), vec3f(0.08, 0.28, 0.88), -scaled), scaled < 0.0);
      alpha = select(0.28, 0.92, wet || abs(scaled) > 0.05);
    } else if (fieldMode == 5) {
      let pressure = textureLoad(mappedPressureField, cell, 0).x; let scale = max(1.0, 10000.0 * u.container.y);
      fill = heatColor(clamp(0.5 + 0.5 * pressure / scale, 0.0, 1.0)); alpha = select(0.22, 0.88, wet);
    } else if (fieldMode == 6) {
      let unrepresented = adaptiveGrid && wet && !hasLiquidPressureDof(cell);
      fill = select(select(vec3f(0.12, 0.20, 0.22), vec3f(0.08, 0.62, 0.50), wet), vec3f(1.0, 0.02, 0.01), unrepresented);
      alpha = select(select(0.10, 0.70, wet), 0.98, unrepresented);
    } else if (fieldMode == 7 && u.environment.w > 0.5) {
      let shape = adaptiveCellVerticalShape(cell, dims);
      let opticalCube = shape.y - shape.x == 1;
      // Gold/cyan is retained cubic optical storage; blue-grey is the merged
      // tall-cell interior. The palette stays categorical so fixed/adaptive
      // A/B runs remain legible even when their layer depths are close.
      fill = select(vec3f(0.15, 0.23, 0.35), select(vec3f(0.40, 0.80, 0.86), vec3f(0.96, 0.68, 0.10), wet), opticalCube);
      alpha = select(select(0.24, 0.70, wet), select(0.56, 0.92, wet), opticalCube);
      // Mark the lowest transition from a tall pressure cell into the cubic
      // optical layer. Coarse cubes span several backing voxels, so only the
      // represented cell's true lower edge receives the accent.
      var belowIsTall = false;
      if (opticalCube && shape.x > 0 && cell.y == shape.x) {
        belowIsTall = !isOpticalCube(vec3i(cell.x, shape.x - 1, cell.z), dims);
      }
      let verticalFraction = fract(samplePosition.y);
      opticalBoundary = select(0.0, 1.0 - smoothstep(0.35, 1.6, verticalFraction / derivative.y), belowIsTall && axis != 3);
    } else if (fieldMode == 8) {
      // Octree materialization bitcasts |u_after-u_before| into the unused
      // second pressure-ownership lane. A dark internal coarse leaf is a
      // pressure-space coverage alarm; the affine prolongation should make
      // hydrostatic and other smooth modes visible throughout the leaf.
      let pressureUpdate = bitcast<f32>(textureLoad(pressureSamples, cell, 0).y);
      fill = heatColor(pressureUpdate / max(u.environment.z, 0.05));
      alpha = select(0.25, 0.92, wet || pressureUpdate > 1e-5);
    } else if (fieldMode == 9 && adaptiveGrid) {
      // Prefer the compact leaf record itself for the octree. This audits the
      // source shared by sparse velocity and surface pages instead of trusting
      // the derived owner texture that draws the structural lines.
      let shape = adaptiveCellVerticalShape(cell, dims);
      let horizontalSize = max(1, shape.z);
      let verticalSize = max(1, shape.y - shape.x);
      var representedSize = max(horizontalSize, verticalSize);var missingCompactOwner=false;var compactArenaFault=false;
      if(octreeGrid&&octreeSurfaceBound()){let row=octreeSurfaceRow(cell);missingCompactOwner=row==SPARSE_SURFACE_INVALID;if(!missingCompactOwner){representedSize=i32(octreeSurfaceLeaves[row].size);}}
      if(octreeGrid&&octreeSurfaceBound()){compactArenaFault=octreeSurfaceArena[3]!=0u;}
      // Normalize to this solver's live hierarchy rather than a hard-coded
      // size-32 tree. For the default octree, size 8 is therefore genuinely
      // blue instead of remaining cyan simply because size 32 is unavailable.
      let maximumRepresentedSize = max(2.0, u.options.w);
      let level = clamp(log2(f32(representedSize)) / log2(maximumRepresentedSize), 0.0, 1.0);
      // The pressure hierarchy is cyan-to-blue. Pink is reserved for the
      // physically finer sparse surface field, avoiding the false impression
      // that every finest pressure leaf is a separate surface resolution.
      let fineColor = vec3f(0.22, 0.68, 0.74);
      let middleColor = vec3f(0.22, 0.68, 0.74);
      let coarseColor = vec3f(0.08, 0.18, 0.48);
      fill = select(mix(middleColor, coarseColor, (level - 0.5) * 2.0), mix(fineColor, middleColor, level * 2.0), level < 0.5);
      alpha = 0.88;if(missingCompactOwner){let hinted=textureLoad(pressureSamples,cell,0).x;fill=select(vec3f(1.0,0.02,0.72),vec3f(0.98,0.64,0.08),hinted<octreeSurfaceParams.shape.x);alpha=0.98;}if(compactArenaFault){fill=vec3f(1.0,0.01,0.01);alpha=0.98;}
    } else if (fieldMode == 10 && adaptiveGrid) {
      // Dedicated sparse-surface audit. Unlike the normal structure and cell
      // scale views, this deliberately exposes whole allocated pages so the
      // true interface shell, its core-page support, the transport halo, and
      // coarse fallback can be distinguished at a glance.
      fill = vec3f(0.05, 0.12, 0.35);
      alpha = 0.78;
      if(octreeSurfaceBound()){
        let row=octreeSurfaceRow(cell);
        if(octreeSurfaceArena[3]!=0u){fill=vec3f(1.0,0.01,0.01);alpha=0.98;}
        else if(row==SPARSE_SURFACE_INVALID){let hinted=textureLoad(pressureSamples,cell,0).x;fill=select(vec3f(1.0,0.02,0.72),vec3f(0.98,0.64,0.08),hinted<octreeSurfaceParams.shape.x);alpha=0.98;}
        else {let state=octreeSurfaceArena[octreeSurfaceParams.offsets0.y+row];let slot=octreeSurfaceArena[octreeSurfaceParams.offsets0.x+row];let resident=(state&1u)!=0u;let core=(state&2u)!=0u;let halo=(state&4u)!=0u;let desired=(state&8u)!=0u;let activated=(state&16u)!=0u;let pageValid=slot<octreeSurfaceParams.shape.y;
          fill=select(fill,vec3f(0.12,0.72,0.82),resident&&halo);fill=select(fill,vec3f(0.42,0.19,0.62),resident&&core);fill=select(fill,vec3f(0.96,0.58,0.08),desired&&!resident);fill=select(fill,vec3f(0.98,0.84,0.10),activated);fill=select(fill,vec3f(1.0,0.03,0.52),core&&abs(octreeSurfacePhi(row,vec3f(cell)+vec3f(0.5)))<=1.5*min(octreeSurfaceParams.cellDt.x,min(octreeSurfaceParams.cellDt.y,octreeSurfaceParams.cellDt.z)));alpha=select(alpha,0.94,resident||desired);
          if(resident!=pageValid){fill=vec3f(1.0,0.01,0.01);alpha=0.98;}
        }
      }else if (sparseSurfaceAvailable()) {
        let factor = f32(sparseSurfaceParams.coarseDims.w);
        let fine3 = local3 * factor;
        let fineCell = vec3i(floor(fine3));
        let state = sparseSurfacePageState(fineCell);
        let resident = (state & 1u) != 0u;
        let core = (state & 2u) != 0u;
        let halo = (state & 4u) != 0u;
        fill = select(fill, vec3f(0.12, 0.72, 0.82), resident && halo);
        fill = select(fill, vec3f(0.42, 0.19, 0.62), resident && core);
        fill = select(fill, vec3f(1.0, 0.03, 0.52), sparseSurfaceCoreSample(fineCell));
        alpha = select(alpha, 0.94, resident);
        if (resident) {
          var finePosition = fine3.xy;
          var fineDerivative = derivative * factor;
          if (axis == 2) { finePosition.x = fine3.z; }
          else if (axis == 3) { finePosition = fine3.xz; }
          let firstFineLine = 1.0 - smoothstep(0.35, 1.15, (0.5 - abs(fract(finePosition.x) - 0.5)) / max(fineDerivative.x, 1e-5));
          let secondFineLine = 1.0 - smoothstep(0.35, 1.15, (0.5 - abs(fract(finePosition.y) - 0.5)) / max(fineDerivative.y, 1e-5));
          line = max(line, max(firstFineLine, secondFineLine));
        }
      }
    } else if (fieldMode == 11 && octreeGrid) {
      let row=octreeSurfaceRow(cell);let audit=octreeFaceAudit(row,vec3f(cell)+vec3f(0.5));
      let load=clamp((f32(audit.count)-6.0)/18.0,0.0,1.0);fill=mix(vec3f(0.08,0.58,0.47),vec3f(0.55,0.22,0.78),load);alpha=0.90;
      if(audit.axisMask!=7u){fill=vec3f(0.98,0.64,0.08);alpha=0.96;}
      if(row==SPARSE_SURFACE_INVALID){let hinted=textureLoad(pressureSamples,cell,0).x;fill=select(vec3f(1.0,0.02,0.72),vec3f(0.98,0.64,0.08),hinted<octreeSurfaceParams.shape.x);alpha=0.98;}
      else if(audit.fault!=0u){fill=vec3f(1.0,0.01,0.01);alpha=0.98;}
    }
  }
  line *= lineFade;
  let gridBody = gridBodySample(representedCell(cell, dims, boundsMin, size, adaptiveGrid, tallGrid));
  if (gridBody.occupied) {
    fill = mix(vec3f(0.96, 0.43, 0.12), vec3f(1.0, 0.78, 0.38), 0.18 * gridBody.selected);
    alpha = 0.97;
    sampleDot = 0.0;
  }
  // Structure remains pressure-topology first. The independently allocated
  // fine surface pages are layered over it below instead of recoloring every
  // finest pressure leaf as though it were fine phi storage.
  var gridLineColor = vec3f(0.03, 0.08, 0.09);
  if ((fieldMode == 0 || fieldMode == 9) && sparseSurfaceAvailable()) {
    let factor=f32(sparseSurfaceParams.coarseDims.w);let fine3=local3*factor;let fineCell=vec3i(floor(fine3));
    if (sparseSurfaceCoreSample(fineCell)) {
      var finePosition=fine3.xy;var fineDerivative=derivative*factor;
      if(axis==2){finePosition.x=fine3.z;}else if(axis==3){finePosition=fine3.xz;}
      let first=1.0-smoothstep(0.35,1.15,(0.5-abs(fract(finePosition.x)-0.5))/max(fineDerivative.x,1e-5));
      let second=1.0-smoothstep(0.35,1.15,(0.5-abs(fract(finePosition.y)-0.5))/max(fineDerivative.y,1e-5));
      let fineLine=max(first,second);let fineColor=vec3f(1.0,0.08,0.55);
      fill=mix(fill,fineColor,0.34);alpha=max(alpha,0.46);line=max(line,fineLine);gridLineColor=fineColor;
    }
  }
  var color = mix(fill, gridLineColor, line);
  color = mix(color, vec3f(0.02, 0.05, 0.06), sampleDot);
  let opticalBoundaryColor = select(vec3f(0.93, 0.93, 0.98), vec3f(1.0, 0.08, 0.55), u.environment.w > 1.5);
  color = mix(color, opticalBoundaryColor, opticalBoundary);
  alpha = max(alpha, max(opticalBoundary, max(line * 0.85, sampleDot * 0.92)));
  return GridSample(color, alpha, gridBody.occupied);
}

fn displayColor(linear: vec3f) -> vec3f {
  let mapped = linear / (linear + vec3f(1.0));
  return pow(max(mapped, vec3f(0.0)), vec3f(1.0 / 2.2));
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let axis = i32(round(u.debug.x));
  if (axis <= 0 || u.gridInfo.w <= 0.5) { discard; }
  let ndc = input.uv * 2.0 - 1.0;
  let origin = u.cameraPosition.xyz;
  let forward = normalize(u.cameraTarget.xyz - origin);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let direction = normalize(forward + right * ndc.x * u.viewport.x / max(u.viewport.y, 1.0) * ${CAMERA_TAN_HALF_FOV} + up * ndc.y * ${CAMERA_TAN_HALF_FOV});
  let size = u.container.xyz;
  let boundsMin = vec3f(-0.5 * size.x, 0.0, -0.5 * size.z);
  let boundsMax = boundsMin + size;
  let dims = vec3f(u.gridInfo.xyz);
  var denominator = direction.z;
  var rayOrigin = origin.z;
  var planeCoordinate = 0.0;
  if (axis == 1) {
    let layer = clamp(floor(u.debug.y * dims.z), 0.0, dims.z - 1.0);
    planeCoordinate = boundsMin.z + (layer + 0.5) * size.z / dims.z;
  } else if (axis == 2) {
    let layer = clamp(floor(u.debug.y * dims.x), 0.0, dims.x - 1.0);
    planeCoordinate = boundsMin.x + (layer + 0.5) * size.x / dims.x;
    denominator = direction.x;
    rayOrigin = origin.x;
  } else {
    let layer = clamp(floor(u.debug.y * dims.y), 0.0, dims.y - 1.0);
    planeCoordinate = boundsMin.y + (layer + 0.5) * size.y / dims.y;
    denominator = direction.y;
    rayOrigin = origin.y;
  }
  if (abs(denominator) <= 1e-5) { discard; }
  let distance = (planeCoordinate - rayOrigin) / denominator;
  let point = origin + direction * distance;
  let inside = all(point >= boundsMin - vec3f(1e-4)) && all(point <= boundsMax + vec3f(1e-4));
  if (distance <= 0.0 || !inside) { discard; }
  let footprint = distance * 1.44 / max(u.viewport.y, 1.0);
  var overlay = gridSample(point, boundsMin, size, axis, footprint);
  if (distance >= nearestBodyDistance(origin, direction) && !overlay.solid) { discard; }
  let horizontalEdgeDistance = min(min(point.x - boundsMin.x, boundsMax.x - point.x), min(point.z - boundsMin.z, boundsMax.z - point.z));
  let grip = select(clamp(1.0 - (boundsMax.y - point.y) / (0.03 * size.y), 0.0, 1.0), clamp(1.0 - horizontalEdgeDistance / (0.035 * min(size.x, size.z)), 0.0, 1.0), axis == 3) * 0.8;
  overlay.color = mix(overlay.color, vec3f(0.51, 0.95, 0.82), grip);
  overlay.alpha = max(overlay.alpha, grip);
  return vec4f(displayColor(overlay.color), overlay.alpha);
}
`;

export class GridOverlayPipeline {
  private pipeline?: GPURenderPipeline;
  private bindGroup?: GPUBindGroup;
  private volume?: GPUTexture;
  private columnBases?: GPUTexture;
  private adaptiveCells?: GPUTexture;
  private velocity?: GPUTexture;
  private pressureSamples?: GPUTexture;
  private divergence?: GPUTexture;
  private mappedPressure?: GPUTexture;
  private sparseSurface?: SparseSurfaceBandGPUSource;
  private adaptiveOctree?: UnifiedOctreeConsumerSource;
  private fallbackSparsePages: GPUBuffer;
  private fallbackSparseParams: GPUBuffer;
  private fallbackSparsePhi: GPUBuffer;
  private fallbackSparseControl: GPUBuffer;

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer
  ) {
    this.fallbackSparsePages = device.createBuffer({ label: "Grid sparse-surface page fallback", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fallbackSparseParams = device.createBuffer({ label: "Grid sparse-surface parameter fallback", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fallbackSparsePhi = device.createBuffer({ label: "Grid sparse-surface phi fallback", size: 4, usage: GPUBufferUsage.STORAGE });
    // This buffer also stands in for one 64-byte OctreeSurfaceLeaf when the
    // adaptive source is absent. Auto-layout validation applies the leaf
    // struct's minimum binding size even though the disabled shader path does
    // not read it.
    this.fallbackSparseControl = device.createBuffer({ label: "Grid sparse-surface control fallback",
      size: OCTREE_SURFACE_LEAF_RECORD_BYTES, usage: GPUBufferUsage.STORAGE });
  }

  async initialize() {
    const shaderModule = this.device.createShaderModule({ label: "Solver grid overlay", code: gridOverlayShader });
    const compilation = await shaderModule.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Solver grid overlay composite",
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.targetFormat,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }
          }
        }]
      },
      primitive: { topology: "triangle-list" }
    });
    this.rebuildBindGroup();
  }

  setVolume(volume: GPUTexture, columnBases: GPUTexture, adaptiveCells: GPUTexture, velocity: GPUTexture, pressureSamples: GPUTexture, divergence: GPUTexture, mappedPressure: GPUTexture) {
    if (this.volume === volume && this.columnBases === columnBases && this.adaptiveCells === adaptiveCells && this.velocity === velocity && this.pressureSamples === pressureSamples && this.divergence === divergence && this.mappedPressure === mappedPressure) return;
    this.volume = volume;
    this.columnBases = columnBases;
    this.adaptiveCells = adaptiveCells;
    this.velocity = velocity;
    this.pressureSamples = pressureSamples;
    this.divergence = divergence;
    this.mappedPressure = mappedPressure;
    this.rebuildBindGroup();
  }

  setSparseSurface(source: SparseSurfaceBandGPUSource | undefined) {
    if (this.sparseSurface === source) return;
    this.sparseSurface = source;
    this.rebuildBindGroup();
  }

  setAdaptiveOctree(source: UnifiedOctreeConsumerSource|undefined){if(source)validateUnifiedOctreeConsumerSource(source);if(this.adaptiveOctree===source)return;this.adaptiveOctree=source;this.rebuildBindGroup();}

  private rebuildBindGroup() {
    if (!this.pipeline || !this.volume || !this.columnBases || !this.adaptiveCells || !this.velocity || !this.pressureSamples || !this.divergence || !this.mappedPressure) return;
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
        { binding: 2, resource: this.volume.createView({ dimension: "3d" }) },
        { binding: 3, resource: this.columnBases.createView() },
        { binding: 4, resource: this.adaptiveCells.createView({ dimension: "3d" }) },
        { binding: 5, resource: this.velocity.createView({ dimension: "3d" }) },
        { binding: 6, resource: this.pressureSamples.createView({ dimension: "3d" }) },
        { binding: 7, resource: this.divergence.createView({ dimension: "3d" }) },
        { binding: 8, resource: this.mappedPressure.createView({ dimension: "3d" }) },
        { binding: 9, resource: this.sparseSurface?.pageTable ?? { buffer: this.fallbackSparsePages } },
        { binding: 10, resource: this.sparseSurface?.params ?? { buffer: this.fallbackSparseParams } },
        { binding: 11, resource: this.sparseSurface?.phi ?? { buffer: this.fallbackSparsePhi } },
        { binding: 12, resource: this.sparseSurface?.control ?? { buffer: this.fallbackSparseControl } },
        { binding: 13, resource: this.sparseSurface?.states ?? { buffer: this.fallbackSparsePages } }
        ,{binding:14,resource:this.adaptiveOctree?.leaves??{buffer:this.fallbackSparseControl}}
        ,{binding:15,resource:this.adaptiveOctree?.surfaceArena??{buffer:this.fallbackSparseControl}}
        ,{binding:16,resource:this.adaptiveOctree?.surfaceParams??{buffer:this.fallbackSparseParams}}
        ,{binding:17,resource:this.adaptiveOctree?.faces??{buffer:this.fallbackSparseControl}}
        ,{binding:18,resource:this.adaptiveOctree?.incidence??{buffer:this.fallbackSparseControl}}
        ,{binding:19,resource:this.adaptiveOctree?.faceControl??{buffer:this.fallbackSparseControl}}
      ]
    });
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView): boolean {
    if (!this.pipeline || !this.bindGroup) return false;
    const pass = encoder.beginRenderPass({
      label: "Solver grid overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }

  destroy() {
    this.fallbackSparsePages.destroy();
    this.fallbackSparseParams.destroy();
    this.fallbackSparsePhi.destroy();
    this.fallbackSparseControl.destroy();
  }
}
