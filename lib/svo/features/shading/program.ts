/** Fused SVO shading program and its packing ABI. The pipeline host owns
 * scheduling and resources; feature extraction must not add GPU dispatches. */
import { PLANAR_BOUNDARY_PATCH_BYTES,planarBoundaryWGSL } from "../../../core/planar-boundary";
import { VOXEL_MATERIAL_IDS } from "../../../core/voxel-scene";
import { cameraApertureShaderLibrary } from "../../../core/webgpu-camera";
import { unifiedLightingShaderLibrary,WATER_OPTICS } from "../../../core/webgpu-lighting";
import { SPARSE_VOXEL_VALID_FIELDS } from "../../../core/webgpu-voxel-debug";
import { svoGBufferWGSL } from "../../contracts/svo-gbuffer";
import { SVO_LIGHT_MAXIMUM_RECORDS,svoLightWGSL } from "../../contracts/svo-light-abi";
import { SVO_MATERIAL_RECORD_STRIDE_BYTES,svoMaterialWGSL } from "../../contracts/svo-material-abi";
import { SVO_CLUSTER_SWEEP_MAXIMUM_POINTS,SVO_PRIMITIVE_RECORD_STRIDE_BYTES,SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,svoPrimitiveWGSL } from "../../contracts/svo-primitive-abi";
import { SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT } from "../../pipeline/svo-render-tuning";
import { octreeLiveSceneLeafPayloadMode,octreeLiveSceneSceneGeometryFormat,sparseBrickBandedLeafCodecWGSL,sparseBrickSceneGeometryCodecWGSL,sparseBrickSceneIdentityCodecWGSL,sparseBrickSceneIdentityWordCodecWGSL } from "../construction/sparse-brick-octree";
import { svoBrickContourWGSL } from "../construction/svo-brick-contour";
import { svoBrickOccupancyWGSL } from "../construction/svo-brick-occupancy";
import { createWebgpuSvoCompactTraversalWGSL } from "../construction/webgpu-svo-compact-hierarchy";
import { createWebgpuSvoWideFanoutTraversalWGSL } from "../construction/webgpu-svo-wide-fanout";
import { type SvoPixelTracePrimaryMode } from "../diagnostics/svo-pixel-trace";
import { createSvoPixelTraceProbeWGSL,type SvoPixelTraceProbeOptions } from "../diagnostics/webgpu-svo-pixel-trace";
import { SVO_CONTACT_VISIBILITY_CONTRACT } from "../lighting-visibility/svo-contact-visibility";
import { createSvoScreenSpaceTraversalWGSL,SVO_SCREEN_SPACE_TERMINATION_CONTRACT,svoLodDescentWGSL } from "../lighting-visibility/svo-screen-space-termination";
import { SVO_VISIBILITY_LIMITS,svoVisibilityRaysWGSL } from "../lighting-visibility/svo-visibility-rays";
import { svoProceduralNoiseWGSL } from "../materials/svo-procedural-material";
import { SVO_SCENE_GLASS_MAXIMUM_PANES } from "../materials/svo-scene-glass";
import { SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES } from "../materials/svo-scene-thick-glass";
import { svoThickGlassWGSL } from "../materials/svo-thick-glass";
import { SVO_THIN_GLASS_RECORD_STRIDE_BYTES,SVO_THIN_GLASS_RECORD_WORDS,svoThinGlassWGSL } from "../materials/svo-thin-glass";
import { svoSurfaceMeshWGSL } from "../primary-visibility/svo-surface-mesh";
import { SVO_BRICK_RASTER_CONTRACT,svoBrickRasterSharedWGSL,svoRasterCoverageOverflowSignalWGSL } from "../primary-visibility/webgpu-svo-brick-raster";
import { SVO_PRIMARY_ENTRY_PREPASS_CONTRACT } from "../primary-visibility/webgpu-svo-primary-entry-prepass";
import { createWebgpuSvoTraversalWGSL } from "../primary-visibility/webgpu-svo-traversal";
import { SVO_CONE_RADIANCE_RECONSTRUCTION_CODES } from "../radiance/definition";
import { svoEnvironmentLightingWGSL } from "../radiance/svo-environment-lighting";
import { SVO_NODE_MIP_LAYOUT } from "../radiance/svo-node-mip-pyramid";
import { svoNodeMipSamplingWGSL } from "../radiance/svo-node-mip-sampling";
import { svoTetrahedralRadianceWGSL } from "../radiance/svo-tetrahedral-radiance";
import { svoTetrahedralRadianceConeCoreWGSL } from "../radiance/svo-tetrahedral-radiance-cone";
import { liveSvoDerivedPageValidityWGSL } from "../radiance/webgpu-svo-live-derived-cache";
import { SVO_FIELD_PROGRAM_BLOCK_WORDS,svoFieldProgramWGSL } from "../scene-publication/svo-field-program";
import { svoFluidCoverageWGSL } from "../scene-publication/svo-fluid-coverage";
import { SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES,SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES,SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK } from "../scene-publication/svo-primitive-candidates";
import { svoPrimitiveMotionWGSL } from "../scene-publication/svo-primitive-motion";
import { svoScenePrimitiveBandReadWGSL } from "../scene-publication/svo-scene-primitive-band";

/**
 * Authored PBR materials one scene may publish.
 *
 * One material per record plus the 32 environment slots, so the acceptance
 * scene's 5 039 records ask for 5 071 — inside this ceiling with 38 % headroom,
 * which is why W3 raises the cluster arena beside it and leaves this alone. The
 * next rung that would cross it is roughly 16x the hero, and the authored record
 * ceiling (16 384) fires first.
 */
export const SVO_DRY_SCENE_MATERIAL_CAPACITY = 8_192;

export const SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_STRIDE_BYTES;

export const SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES =
  SVO_SCENE_GLASS_MAXIMUM_PANES * SVO_THIN_GLASS_RECORD_STRIDE_BYTES;

export const alignDrySceneArenaBytes = (value: number): number => Math.ceil(value / 256) * 256;

/**
 * How many aggregate clusters one scene may publish.
 *
 * A block is a 192-byte slot (the tapered sweep carries an eight-point
 * polyline), written once per publication rather than per frame.
 *
 * Raised from 1 024 with the record ceilings W0 took from 4 096 to 16 384. The
 * old value was chosen when it was "well above anything the record budget can
 * reach", and that stopped being true: `hero-garden-hose-x10` spends **948** of
 * 1 024 blocks at its acceptance multiplier, so the 10x rung only fit at all
 * because that scene is authored deliberately cluster-light — one miniature
 * bonsai per nine stands. Any species mix nearer the hero's own 41.5 % aggregate
 * share could not reach 10x, which would have made the ceiling, rather than the
 * renderer, the thing the acceptance gate measured. Four thousand and ninety-six
 * blocks is 786 KB reserved once, against a scene arena that already carries
 * three quarters of a megabyte of material and candidate space and a coverage
 * arena three orders of magnitude larger.
 */
export const SVO_DRY_SCENE_CLUSTER_CAPACITY = 4_096;

export const SVO_DRY_SCENE_CLUSTER_BLOCK_BYTES =
  SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS * Uint32Array.BYTES_PER_ELEMENT;

export const SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_CLUSTER_CAPACITY * SVO_DRY_SCENE_CLUSTER_BLOCK_BYTES;


/**
 * How many field-program tapes one scene may publish.
 *
 * Two hundred and fifty-six, and deliberately an order of magnitude under the
 * aggregate arena beside it, because the two kinds are used at opposite
 * densities. An aggregate is authored once per *record* — the hero garden
 * publishes 208 of them against 501 records — while a field program is authored
 * once per *species*: one boulder tape is instanced across every stone in a set,
 * and the whole point of the kind is that adding detail to it does not add
 * records. A scene that genuinely wanted a distinct tape per record would be
 * saying it has no shared shape language at all, which is the case this kind
 * exists to replace.
 *
 * A block is 528 bytes, so the region is 132 KB reserved once, against a scene
 * arena that already carries three quarters of a megabyte of material and
 * candidate space.
 */
export const SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY = 256;

export const SVO_DRY_SCENE_FIELD_PROGRAM_BLOCK_BYTES =
  SVO_FIELD_PROGRAM_BLOCK_WORDS * Uint32Array.BYTES_PER_ELEMENT;

export const SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES =
  SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY * SVO_DRY_SCENE_FIELD_PROGRAM_BLOCK_BYTES;


/**
 * One stable renderer-owned storage allocation for every authored scene record.
 *
 * Written as a running offset so adding a fixed publication region does not
 * require hand-updating every following byte offset.
 */
export const drySceneArenaRegions = (() => {
  let offsetBytes = 0;
  const region = (sizeBytes: number): number => {
    const start = offsetBytes;
    offsetBytes = alignDrySceneArenaBytes(start + sizeBytes);
    return start;
  };
  const materialOffsetBytes = region(SVO_DRY_SCENE_MATERIAL_ARENA_SIZE_BYTES);
  const primitiveOffsetBytes = region(SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES);
  const glassOffsetBytes = region(SVO_DRY_SCENE_GLASS_ARENA_SIZE_BYTES);
  const clusterOffsetBytes = region(SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES);
  const fieldProgramOffsetBytes = region(SVO_DRY_SCENE_FIELD_PROGRAM_ARENA_SIZE_BYTES);
  return {
    materialOffsetBytes, primitiveOffsetBytes, glassOffsetBytes, clusterOffsetBytes, fieldProgramOffsetBytes,
    sizeBytes: offsetBytes,
  };
})();


export const SVO_DRY_SCENE_ARENA_LAYOUT = Object.freeze({
  materialOffsetBytes: drySceneArenaRegions.materialOffsetBytes,
  primitiveOffsetBytes: drySceneArenaRegions.primitiveOffsetBytes,
  glassOffsetBytes: drySceneArenaRegions.glassOffsetBytes,
  /** Fixed-capacity aggregate parameter blocks, addressed by a record's word-13 reference. */
  clusterOffsetBytes: drySceneArenaRegions.clusterOffsetBytes,
  /** Fixed-capacity field-program tape blocks, addressed by a record's word-13 reference. */
  fieldProgramOffsetBytes: drySceneArenaRegions.fieldProgramOffsetBytes,
  sizeBytes: drySceneArenaRegions.sizeBytes,
} as const);


/**
 * Word offset of aggregate cluster block `index` in the scene arena.
 *
 * Same convention as the terrain reference and for the same reason: a shader
 * handed this number reads the block with no further indirection, so the
 * reference is the offset rather than an index into a table that would have to
 * exist first.
 */
/**
 * Word offsets inside a cluster's arena block.
 *
 * Declared once and read by the packer, the CPU resolver and the shader's own
 * decode below, because those are three transcriptions of the same layout and
 * the failure mode when one of them drifts is a field that renders as a
 * different field with no error anywhere. The order matches
 * `struct SvoClusterPacking` in the shared ABI so the decode is a straight run
 * down the block.
 */
export const CLUSTER_BLOCK_FIELD_WORD = 0;

export const CLUSTER_BLOCK_SEED_WORD = 1;

export const CLUSTER_BLOCK_COUNT_WORD = 2;

export const CLUSTER_BLOCK_SMOOTH_RADIUS_WORD = 3;

export const CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD = 4;

export const CLUSTER_BLOCK_LOBE_SPAN_WORD = 8;

/**
 * Eight `vec4f`, so the shader reads the polyline as vectors rather than
 * scalars. Sixteen-word aligned, which is why the three words above it stop at
 * eleven and the block leaves four unused.
 */
export const CLUSTER_BLOCK_POINTS_WORD = 16;


/** materialPublication.w flags shared by the direct and derived-lighting paths. */
export const SVO_DRY_VISIBILITY_FLAGS = Object.freeze({
  exactContact: 1 << 0,
  exactShadow: 1 << 1,
  coneLightingRequested: 1 << 2,
  ambientOcclusion: 1 << 3,
  globalIllumination: 1 << 4,
  globalIlluminationOcclusion: 1 << 5,
  globalIlluminationRequested: 1 << 6,
  silhouetteRefinement: 1 << 7,
  flatVoxelNormals: 1 << 8,
} as const);


/** How the stable node-mip address plan proves that an atlas sample is current. */
export const SVO_DRY_NODE_MIP_PUBLICATION_MODE = Object.freeze({
  unavailable: 0,
  matchingStructuralGeneration: 1,
  pageValidity: 2,
} as const);


/** Reversed-Z near plane the dry pass writes device depth against. */
export const SVO_DRY_SCENE_REVERSED_Z_NEAR_M = 0.01;

/**
 * fract(lod) width of the transition band in which the cone marcher blends the
 * two bracketing mip levels; below the band a single fine-level fetch suffices.
 * The concentric-ring artifact came from C0 discontinuity at integer LOD
 * switches, not from lack of full-range blending, so the band's blend weight
 * ramps 0 at the band start to 1 at fract==1 (where it equals the next level's
 * band-start value): coverage stays continuous at both band edges while ~70%
 * of steps skip the second atlas fetch and its directory/page-cache work.
 * Measured (M1 Max, garden 1280x720): full-range blending cost scale-1
 * 40.6 ms / scale-0.5 16.0 ms; width 0.3 recovers most of the two-fetch
 * regression with no visible banding (0.5 measured within noise of 0.3).
 */
export const SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH = 0.3;

/** Bound direct-light work independently from the producer's 32-record capacity. */
export const SVO_DRY_SCENE_MAX_SHADED_LIGHTS = 8;

/** Two fixed shape samples are stable across frames and keep total visibility work bounded. */
export const SVO_DRY_SCENE_AREA_LIGHT_SAMPLES = 2;

/** AO sample ceiling, independent of camera activity. */
export const SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES = 4;


/**
 * Step ceiling for the fluid optical-depth march.
 *
 * The march is clipped to the coverage box and its step is the cone footprint,
 * so a widening shadow cone crossing the box needs a logarithmic number of
 * steps and typically finishes in well under ten. The ceiling only bounds the
 * degenerate case — a near-pencil cone traversing the box's long diagonal —
 * where 32 steps still resolve the volume at a third of a texel per step.
 */
export const SVO_DRY_FLUID_MARCH_STEPS = 32;


/**
 * Cone directions the fluid contact term averages.
 *
 * Two, not the solid term's four. The solid term needs four because scenery
 * occlusion is high-frequency — an edge either blocks a direction or does not,
 * and too few directions band. Coverage is a low-frequency field read through a
 * mip chain, so its hemisphere estimate is already smooth and the extra pair
 * moves the result by less than a quantization step of the stored byte lane.
 */
export const SVO_DRY_CONTACT_FLUID_SAMPLES = 2;


// Chrome's WGSL frontend requires the renderer-supplied adapter declaration
// before the shared trace body which calls it. Naga accepts the forward call,
// so keep the composition order explicit here and covered by integration tests.
export const SVO_VISIBILITY_TRACE_MARKER = "fn svoTraceVisibility(";

export const svoVisibilityTraceOffset = svoVisibilityRaysWGSL.indexOf(SVO_VISIBILITY_TRACE_MARKER);

export const svoVisibilityPreludeWGSL = svoVisibilityRaysWGSL.slice(0, svoVisibilityTraceOffset);

export const svoVisibilityTraceWGSL = svoVisibilityRaysWGSL.slice(svoVisibilityTraceOffset);


/**
 * Structural fields an SVO-accelerated primary ray needs before it may leave the camera.
 * Primary and secondary traversal both refuse to consume the SVO until the
 * producer has published this live-scene field set. A producer that allocates
 * the structural source but never finalizes its current revision renders every
 * accelerated surface as a miss; analytic glass and rigid bodies keep drawing.
 *
 * The set is the tree, per-voxel identity, and scene geometry. Identity decides
 * solidity and supplies the baked normal; smooth surface reconstruction reads
 * the geometry lane's coverage fraction to place its tangent plane. A lagging
 * geometry publication must therefore withhold the SVO instead of mixing a new
 * identity with old sub-voxel depths.
 *
 * Today's producer finalizes all three in one pass
 * ({@link OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS}), so this requirement changes no
 * frame that renders now. It changes which *future* publication is legal.
 */
export const SVO_DRY_SCENE_REQUIRED_VALID_FIELDS =
  SPARSE_VOXEL_VALID_FIELDS.topology
  | SPARSE_VOXEL_VALID_FIELDS.sceneGeometry
  | SPARSE_VOXEL_VALID_FIELDS.materialOwner;


/** Feature toggles for the cone-traced node-mip marcher. Production enables every optimization. */
export interface SvoDryConeMarcherOptions {
  /** Branchless Morton bit-spread encode instead of the 21-iteration reference loop. */
  branchlessMorton?: boolean;
  /** Restrict the directory binary search to the queried level's contiguous row range. */
  rangedDirectorySearch?: boolean;
  /** Substitute provably zero coverage without fetching when inside a certified empty region. */
  emptySpaceElision?: boolean;
  /**
   * Accumulate an optical path length through evolving fluid alongside solid
   * coverage. Water is dielectric, not an occluder: turning its coverage into
   * opacity would render a pond as a hole. The cone therefore reports metres of
   * water traversed and leaves the wavelength-dependent attenuation to the
   * caller, which is the same Beer-Lambert term the raster composite already
   * applies along the view ray.
   *
   * Only shadow cones pay for the fetch. Ambient-occlusion cones pass a zero
   * surface normal and skip it, so contact darkening keeps its exact cost.
   */
  fluidCoverage?: boolean;
  /** Resolve a virtual page with one r32uint texture load before directory fallback. */
  directPageTable?: boolean;
}


/**
 * Morton/find/`dryNodeMipAt`/`dryConeVisibility` marcher block shared by the
 * production dry shader and the A/B cone benchmark. Every variant is bit-exact:
 * optimizations may only change how a value is computed, never the value.
 * Requires bindings/declarations named `dry` (DryParams), `publicationState`,
 * `nodeMipAtlas`, `nodeMipSampler`, `nodeMipDirectory`, `nodeMipPageTable`,
 * `nodeMipPageValidity`, and the `svoNodeMipSamplingWGSL` library.
 */
export function createSvoDryConeMarcherWGSL(options: SvoDryConeMarcherOptions = {}): string {
  const morton = options.branchlessMorton
    ? /* wgsl */ `fn dryNodeMipSpreadMortonBits(value:vec3u)->vec3u{
  var spread=value;
  spread=(spread|(spread<<vec3u(16u)))&vec3u(0xff0000ffu);
  spread=(spread|(spread<<vec3u(8u)))&vec3u(0x0f00f00fu);
  spread=(spread|(spread<<vec3u(4u)))&vec3u(0xc30c30c3u);
  spread=(spread|(spread<<vec3u(2u)))&vec3u(0x49249249u);
  return spread;
}
fn dryNodeMipMorton(coordinate:vec3u)->vec2u{
  let masked=coordinate&vec3u(0x1fffffu);
  let low=dryNodeMipSpreadMortonBits(vec3u(masked.x&0x7ffu,masked.y&0x7ffu,masked.z&0x3ffu));
  let high=dryNodeMipSpreadMortonBits(vec3u(masked.x>>11u,masked.y>>11u,masked.z>>10u));
  return vec2u(low.x|(low.y<<1u)|(low.z<<2u),(high.x<<1u)|(high.y<<2u)|high.z);
}`
    : /* wgsl */ `fn dryNodeMipMorton(coordinate:vec3u)->vec2u{
  var result=vec2u(0u);for(var bit=0u;bit<21u;bit+=1u){for(var axis=0u;axis<3u;axis+=1u){let outputBit=bit*3u+axis;let value=(coordinate[axis]>>bit)&1u;if(outputBit<32u){result.x|=value<<outputBit;}else{result.y|=value<<(outputBit-32u);}}}return result;
}`;
  // Directory rows are sorted by (level, morton), so each level occupies one
  // contiguous run; a lower_bound over that run equals the full-range result.
  // Constant vector indexing only: a dynamically indexed uniform array trips a
  // slow-path Tint/Metal transform that taxes the whole fragment shader.
  const levelStart = options.rangedDirectorySearch
    ? /* wgsl */ `fn dryNodeMipLevelStart(level:u32)->u32{
  let clamped=min(level,11u);
  let word=select(select(dry.nodeMipLevelStart[0],dry.nodeMipLevelStart[1],clamped>=4u),dry.nodeMipLevelStart[2],clamped>=8u);
  let lane=clamped&3u;
  return select(select(select(word.x,word.y,lane==1u),word.z,lane==2u),word.w,lane==3u);
}
`
    : "";
  const searchRange = options.rangedDirectorySearch
    ? /* wgsl */ `var low=dryNodeMipLevelStart(level);var high=select(dry.nodeMip.y,dryNodeMipLevelStart(level+1u),level<11u);`
    : /* wgsl */ `var low=0u;var high=dry.nodeMip.y;`;
  const directPageTable = options.directPageTable ? /* wgsl */ `fn dryNodeMipDirectLevelZ(level:u32)->u32{
  let clamped=min(level,11u);
  let word=select(select(dry.nodeMipDirectLevelZ[0],dry.nodeMipDirectLevelZ[1],clamped>=4u),dry.nodeMipDirectLevelZ[2],clamped>=8u);
  let lane=clamped&3u;
  return select(select(select(word.x,word.y,lane==1u),word.z,lane==2u),word.w,lane==3u);
}
fn dryNodeMipDirectFind(level:u32,coordinate:vec3u)->u32{
  if(dry.nodeMipDirect.w==0u||level>=dry.nodeMip.z){return 0xffffffffu;}
  let zStart=dryNodeMipDirectLevelZ(level);
  let zEnd=select(dry.nodeMipDirect.z,dryNodeMipDirectLevelZ(level+1u),level+1u<dry.nodeMip.z);
  if(coordinate.x>=dry.nodeMipDirect.x||coordinate.y>=dry.nodeMipDirect.y||coordinate.z>=zEnd-zStart){return 0xffffffffu;}
  let encoded=textureLoad(nodeMipPageTable,vec3i(vec3u(coordinate.x,coordinate.y,zStart+coordinate.z)),0).x;
  return select(0xffffffffu,encoded-1u,encoded!=0u);
}
` : "";
  const directFind = options.directPageTable
    ? /* wgsl */ `if(dry.nodeMipDirect.w!=0u){return dryNodeMipDirectFind(level,coordinate);}`
    : "";
  const pageOrigin = options.directPageTable
    ? /* wgsl */ `if(dry.nodeMipDirect.w!=0u){let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);let atlasPages=max(dry.nodeMipAtlas.xyz/vec3u(physical),vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));*pageCache=DryNodeMipPageCache(pageCoordinate,level,atlasPage*physical,dry.nodeMip.x,1u,pageIndex,0u);}else{let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);*pageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,0u);}`
    : /* wgsl */ `let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);*pageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,0u);`;
  const zeroRegion = options.emptySpaceElision
    ? /* wgsl */ `struct DryConeZeroRegion{minimum:vec3f,maximum:vec3f,valid:u32}
fn dryConeZeroRegionAt(position_m:vec3f,level:u32,pageCache:ptr<function,DryNodeMipPageCache>)->DryConeZeroRegion{
  // A cached page key is only trustworthy once its coordinate is recomputed in
  // range: the dryNodeMipAt out-of-range early return leaves the cache stale.
  let levelWidth=dry.mapping.cellSize*exp2(f32(level));
  let levelVoxel=(position_m-dry.nodeMipOrigin.xyz)/levelWidth;
  let levelPageFloor=floor(levelVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(levelPageFloor<vec3f(0.0))||any(levelPageFloor>=vec3f(2097152.0))
    ||(*pageCache).generation!=dry.nodeMip.x||(*pageCache).level!=level||any((*pageCache).coordinate!=vec3u(levelPageFloor))||(*pageCache).resident!=0u){
    return DryConeZeroRegion(vec3f(0.0),vec3f(0.0),0u);
  }
  // Non-resident page: no page means no atlas content, so every sample whose
  // trilinear support sits inside this page extent is exactly zero.
  var region=DryConeZeroRegion(
    dry.nodeMipOrigin.xyz+levelPageFloor*f32(SVO_NODE_MIP_INTERIOR_SIZE)*levelWidth,
    dry.nodeMipOrigin.xyz+(levelPageFloor+vec3f(1.0))*f32(SVO_NODE_MIP_INTERIOR_SIZE)*levelWidth,1u);
  let coarseLevel=min(level+2u,dry.nodeMip.z-1u);
  if(coarseLevel>level){
    // Directory-only coarse upgrade (no texture fetch): a non-resident coarse
    // page has no resident descendants via the ancestor-residency chain, so the
    // whole coarse page extent is zero.
    let coarseWidth=dry.mapping.cellSize*exp2(f32(coarseLevel));
    let coarsePageFloor=floor((position_m-dry.nodeMipOrigin.xyz)/(coarseWidth*f32(SVO_NODE_MIP_INTERIOR_SIZE)));
    if(dryNodeMipFind(coarseLevel,vec3u(coarsePageFloor))==0xffffffffu){
      region=DryConeZeroRegion(
        dry.nodeMipOrigin.xyz+coarsePageFloor*f32(SVO_NODE_MIP_INTERIOR_SIZE)*coarseWidth,
        dry.nodeMipOrigin.xyz+(coarsePageFloor+vec3f(1.0))*f32(SVO_NODE_MIP_INTERIOR_SIZE)*coarseWidth,1u);
    }
  }
  return region;
}
`
    : "";
  // Both variants march the identical continuous-LOD cone: the step width
  // follows the continuous cone diameter, and coverage is C0-continuous in
  // lod. A single floor(lod) fetch with a floored step width made accumulated
  // opacity jump wherever floor(lod) incremented along the cone, which
  // rendered as concentric isodistance rings around point lights. Continuity
  // is restored by blending the two bracketing mip levels — but only inside
  // the trailing fract(lod) transition band (SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH
  // wide): the blend weight ramps from 0 at the band start to 1 at
  // fract(lod)==1, where the blended value equals the next level's band-start
  // value, so coverage is continuous at both band edges. Below the band a
  // single fine-level fetch suffices, which skips the second atlas fetch and
  // its directory/page-cache work on most steps (full-range blending doubled
  // per-step fetches everywhere and cost +35% frame time).
  //
  // surfaceNormal (zero only for free-space/standalone benchmark cones)
  // marks the receiver's tangent plane at the march origin: coverage whose
  // trilinear support still straddles that plane is the receiver's own
  // voxelized surface, and accumulating it self-shadows in bands that track
  // the sub-voxel phase of the analytic surface (terrain height isolines
  // rendered as concentric rings, latitude bands on mushroom caps). Each
  // sample's coverage is therefore scaled by its plane clearance over the
  // sample's own support width, ramping back to full occlusion by 24 fine
  // voxels of marched distance so genuine distant blockers keep their shadows.
  // Receiver cones (surfaceNormal set) suppress their own coverage. Shadow
  // cones additionally refine their step width geometrically as
  // the march approaches its endpoint and fade the trailing 1.5 diameters:
  // every cone toward a light converges on the same emitter neighbourhood, so
  // with diameter-sized steps the number of samples landing inside geometry
  // near the endpoint (the receiver's distance modulo the step size) is
  // quantized, which rendered as concentric rings around point lights. Zeno
  // steps shrink the per-sample opacity near the endpoint until the banding
  // amplitude vanishes while the .25-voxel step floor bounds the extra work.
  // Water is measured, not composited: `stepWidth * coverage` is the length of
  // this step that lies inside liquid, so summing it over the march yields the
  // path length a light ray travels through water. No selfWeight here — a
  // receiver standing in a pond really is under water, and suppressing the
  // first samples would erase exactly the shading that makes it read as wet.
  const fluidAccumulation = options.fluidCoverage ? /* wgsl */ `if(shadowCone){fluidDepth_m+=stepWidth*svoFluidCoverageAt(fluidCoverageVolume,nodeMipSampler,dry.fluidCoverage,position,svoFluidCoverageLod(diameter,fluidTexel_m));}` : "";
  const fluidPrologue = options.fluidCoverage ? /* wgsl */ `var fluidDepth_m=0.0;let fluidTexel_m=max(dry.fluidCoverage.texelSize_m.x,max(dry.fluidCoverage.texelSize_m.y,dry.fluidCoverage.texelSize_m.z));` : "";
  const coneMiss = options.fluidCoverage ? "DryConeVisibility(1.0,0u,0.0)" : "DryConeVisibility(1.0,0u)";
  const coneResult = options.fluidCoverage
    ? "DryConeVisibility(clamp(transmittance,0.0,1.0),1u,max(fluidDepth_m,0.0))"
    : "DryConeVisibility(clamp(transmittance,0.0,1.0),1u)";
  const coneStruct = options.fluidCoverage
    ? "struct DryConeVisibility{transmittance:f32,valid:u32,fluidDepth_m:f32}"
    : "struct DryConeVisibility{transmittance:f32,valid:u32}";
  const stepWidthExpression = /* wgsl */ `let remaining=maximumDistance_m-distance;let stepWidth=min(diameter,remaining);`;
  const selfCoverageWeight = /* wgsl */ `var selfWeight=1.0;if(shadowCone){selfWeight=max(clamp(dot(position-origin_m,surfaceNormal)/(1.5*diameter)-1.0,0.0,1.0),clamp((distance-12.0*minimumVoxel)/(12.0*minimumVoxel),0.0,1.0))*clamp(remaining/(1.5*diameter),0.0,1.0);}`;
  const bandStart = 1 - SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH;
  const blendWeightExpression = /* wgsl */ `let blendWeight=clamp((fract(lod)-${bandStart})*${(1 / SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH).toFixed(8)},0.0,1.0);`;
  const coarseBlendedCoverage = /* wgsl */ `var coverage=max(lookup.sample.solidMean,lookup.sample.solidMaximum*.15);if(blendWeight>0.0){let lookupCoarse=dryNodeMipAt(position,lod+1.0,&pageCacheCoarse);if(lookupCoarse.valid==0u){return ${coneMiss};}coverage=mix(coverage,max(lookupCoarse.sample.solidMean,lookupCoarse.sample.solidMaximum*.15),blendWeight);}`;
  const blendedCoverage = /* wgsl */ `let conservativeCoverage=selfWeight*coverage;let alpha=svoNodeMipCoverageOpacity(conservativeCoverage,stepWidth/diameter);transmittance*=1.0-alpha;${fluidAccumulation}`;
  // Phase B: light-anchored geometric ladder over the far half of the march.
  // Phase A's sample grid is anchored at the receiver, so how many samples
  // land inside the mip-smeared coverage around the emitter (lamp globe, head,
  // pole) aliases with the receiver's distance modulo the local step width,
  // which rendered as concentric rings around point lights and latitude bands
  // on nearby caps. The ladder offsets are measured FROM the march endpoint
  // (a fixed clearance off the emitter surface), so its sample positions are
  // world-locked around the light for every receiver: coverage near the light
  // then varies only smoothly with direction and the rings vanish. Ordering is
  // nearest-to-light first so a shared budget exhaustion drops mid-air rungs.
  const emitterLadderWGSL = /* wgsl */ `
  if(anchored){var emitterOffset=minimumVoxel*3.0;
  for(var rung=0u;rung<48u&&budget>0u&&emitterOffset<maximumDistance_m-phaseSplit&&transmittance>.005;rung+=1u){budget-=1u;
    let distance=maximumDistance_m-emitterOffset;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);let remaining=emitterOffset;let stepWidth=emitterOffset*.5;let position=origin_m+direction*distance;
    let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}
    ${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}emitterOffset*=1.5;}}
`;
  // The elision variant keeps the identical march (step distances, stepIndex
  // sequence, termination) and only replaces the
  // fine-level fetch whose trilinear support is provably inside a zero region
  // with the arithmetically identical zero sample: max(0,0*.15)=0 contributes
  // nothing to the blend.
  const visibility = options.emptySpaceElision
    ? /* wgsl */ `fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->DryConeVisibility{
  if(!dryNodeMipReady()){return ${coneMiss};}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;${fluidPrologue}var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;var budget=clamp(dry.tuningCounts0.y,1u,48u);let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  var zeroRegion=DryConeZeroRegion(vec3f(0.0),vec3f(0.0),0u);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}
    let position=origin_m+direction*distance;let level=min(u32(max(floor(lod),0.0)),dry.nodeMip.z-1u);
    // Tap texels lie within 1.5 level-voxels of the sample position, so the
    // whole trilinear support footprint sits inside this conservative box.
    let supportRadius=1.5*dry.mapping.cellSize*exp2(f32(level));
    var lookup=DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);
    if(zeroRegion.valid==0u||any(position-supportRadius<zeroRegion.minimum)||any(position+supportRadius>zeroRegion.maximum)){
      lookup=dryNodeMipAt(position,lod,&pageCache);
      // Establish (or replace) a region only from a non-resident page and only
      // once the march has left the current region entirely: re-deriving the
      // same box would repeat its directory probe for nothing.
      if(pageCache.resident==0u&&(zeroRegion.valid==0u||any(position<zeroRegion.minimum)||any(position>zeroRegion.maximum))){
        zeroRegion=dryConeZeroRegionAt(position,level,&pageCache);
      }
    }
    if(lookup.valid==0u){return ${coneMiss};}
    // The zero region certifies levels at or below its establishment level
    // only, so the in-band coarse bracketing fetch always misses the region
    // and goes through the coarse page cache.
    ${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}distance+=max(stepWidth,minimumVoxel*.25);}${emitterLadderWGSL}
  return ${coneResult};
}`
    : /* wgsl */ `fn dryConeVisibility(origin_m:vec3f,direction:vec3f,aperture:f32,maximumDistance_m:f32,surfaceNormal:vec3f,anchored:bool)->DryConeVisibility{
  if(!dryNodeMipReady()){return ${coneMiss};}let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let tangent=tan(aperture*.5);var distance=minimumVoxel*.75;var transmittance=1.0;${fluidPrologue}var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);var pageCacheCoarse=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let shadowCone=dot(surfaceNormal,surfaceNormal)>.25;var budget=clamp(dry.tuningCounts0.y,1u,48u);let phaseSplit=select(maximumDistance_m,maximumDistance_m*.5,anchored);
  for(var stepIndex=0u;stepIndex<48u&&budget>0u&&distance<phaseSplit&&transmittance>.005;stepIndex+=1u){budget-=1u;let diameter=max(minimumVoxel,2.0*distance*tangent);let lod=svoNodeMipLod(diameter,minimumVoxel);${stepWidthExpression}let position=origin_m+direction*distance;let lookup=dryNodeMipAt(position,lod,&pageCache);if(lookup.valid==0u){return ${coneMiss};}${selfCoverageWeight}${blendWeightExpression}${coarseBlendedCoverage}${blendedCoverage}distance+=max(stepWidth,minimumVoxel*.25);}${emitterLadderWGSL}
  return ${coneResult};
}`;
  return /* wgsl */ `${liveSvoDerivedPageValidityWGSL}
struct DryNodeMipLookup{sample:SvoNodeMipSample,valid:u32}
struct DryNodeMipPageCache{coordinate:vec3u,level:u32,pageOrigin:vec3u,generation:u32,resident:u32,pageIndex:u32,blackRadiance:u32}
${morton}
fn dryNodeMipCompare(entry:SvoNodeMipDirectoryEntry,level:u32,morton:vec2u)->i32{
  if(entry.level<level){return -1;}if(entry.level>level){return 1;}if(entry.mortonHigh<morton.y){return -1;}if(entry.mortonHigh>morton.y){return 1;}if(entry.mortonLow<morton.x){return -1;}if(entry.mortonLow>morton.x){return 1;}return 0;
}
${directPageTable}${levelStart}fn dryNodeMipFind(level:u32,coordinate:vec3u)->u32{
  if(level>=dry.nodeMip.z||dry.nodeMip.y==0u){return 0xffffffffu;}${directFind}let morton=dryNodeMipMorton(coordinate);${searchRange}
  for(var iteration=0u;iteration<24u&&low<high;iteration+=1u){let middle=low+(high-low)/2u;let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,middle);let comparison=dryNodeMipCompare(entry,level,morton);if(comparison<0){low=middle+1u;}else{high=middle;}}
  if(low>=dry.nodeMip.y){return 0xffffffffu;}let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,low);if(entry.generation!=dry.nodeMip.x||dryNodeMipCompare(entry,level,morton)!=0){return 0xffffffffu;}return low;
}
fn dryNodeMipReady()->bool{let generationReady=dry.nodeMip.w==${SVO_DRY_NODE_MIP_PUBLICATION_MODE.pageValidity}u||dry.nodeMip.x==dryPublicationWord(2u);return dry.nodeMip.w!=0u&&dry.nodeMip.x!=0u&&generationReady&&dry.nodeMip.y>0u&&dry.nodeMip.z>0u;}
fn dryNodeMipPageValid(pageIndex:u32)->bool{
  let dimensions=textureDimensions(nodeMipPageValidity);return svoDerivedPageValidityResident(dimensions,pageIndex)&&textureLoad(nodeMipPageValidity,svoDerivedPageValidityTexel(dimensions,pageIndex),0).x!=0u;
}
// Finest level that owns an opacity page (SVO_OPACITY_LEVEL_FLOOR). The base
// level is 81 % of the pyramid's pages at refinement depth 3 and is one texel
// per finest voxel, so a world fine enough to need it stores a resolution no
// cone ever asks for. Zero is the pyramid that shipped, and there this clamp is
// the identity — which is why a reference-leaf frame is bit-exact across it.
fn dryNodeMipOpacityLevelFloor()->u32{return min(dry.nodeMipAtlas.w,dry.nodeMip.z-1u);}
fn dryNodeMipAt(position_m:vec3f,lodIn:f32,pageCache:ptr<function,DryNodeMipPageCache>)->DryNodeMipLookup{
  let level=min(max(u32(max(floor(lodIn),0.0)),dryNodeMipOpacityLevelFloor()),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));let virtualVoxel=(position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}let pageCoordinate=vec3u(pageFloor);
  if((*pageCache).generation!=dry.nodeMip.x||(*pageCache).level!=level||any((*pageCache).coordinate!=pageCoordinate)){
    *pageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){${pageOrigin}}
  }
  if((*pageCache).resident==0u){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),1u);}
  if(!dryNodeMipPageValid((*pageCache).pageIndex)){return DryNodeMipLookup(SvoNodeMipSample(0.0,0.0,0.0,0.0),0u);}
  let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);return DryNodeMipLookup(svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,(*pageCache).pageOrigin,local),1u);
}
${zeroRegion}${coneStruct}
${visibility}`;
}


/**
 * Per-axis resolution scale for the cone-lighting prepass; 1 keeps the inline
 * path.
 *
 * Camera-state budgets remain compile-time-free. The renderer retains and
 * prewarms the configured lighting rates (0.25 performance and 0.5 quality),
 * so a camera-state tier can switch without compiling a Metal shader when
 * motion begins.
 */
export type SvoConeLightingScale = 1 | 0.5 | 0.25 | 0.125;


/** Compile-time static traversal experiment. Hybrid preserves the shipping fallback semantics. */
/**
 * `raster-primary` keeps canonical-parametric traversal for every secondary
 * ray and replaces only the full-screen primary megakernel with a hardware
 * rasterization of resident brick proxies plus a bounded in-brick DDA
 * (docs/SVO_RASTER_PRIMARY_HANDOFF.md).
 */
export type SvoDryTraversalMode = "hybrid" | "canonical" | "canonical-parametric" | "compact" | "wide" | "raster-primary";


export const SVO_DRY_TRAVERSAL_MODES: readonly SvoDryTraversalMode[] = Object.freeze([
  "hybrid", "canonical", "canonical-parametric", "compact", "wide", "raster-primary",
]);


/** Compile-time 8^3 leaf acceleration experiment; off preserves the baseline shader. */
export type SvoBrickOccupancyMode = "off" | "bounds" | "macro" | "macro-hdda";


/** Explicit shading topology: production never changes topology from tuning state. */
export type SvoDryShadingPath = "inline" | "split";


/**
 * Dawn occupancy experiments. Every arm remains independently selectable for
 * controlled A/Bs; only the safe reduced-split diagnostic diet is enabled by
 * the renderer default.
 */
export interface SvoDryOptimizationExperiments {
  /** Compile a guarded opaque, single-directional-light cone closure alongside the generic closure. */
  readonly specializedDeferredLighting?: boolean;
  /** Internal shader variant; selected only with a matching publication and ready cone hierarchy. */
  readonly opaqueDirectionalCones?: boolean;
  /** Resolve current-frame radiance and exact fallbacks in one draw; false retains the A/B reference. */
  readonly singlePassReconstruction?: boolean;
  /** Cached opaque voxel boundary triangles; unavailable publications fail closed. */
  readonly surfaceMesh?: boolean;
  /** Disable only for paired performance/image comparisons. */
  readonly surfaceMeshCulling?: boolean;
  /** Optional diagnostic budget; overflow retains exact ray rendering. */
  readonly surfaceMeshMaxBytes?: number;
  /** Persistent level-0 voxel visibility for directional light slot zero. */
  readonly voxelLightCache?: boolean;
  /** Bounded exact-identity receiver search for sub-prepass-pixel surfaces. */
  readonly edgeReceiverRecovery?: boolean;
  /** Trace incoherent 2x2 receivers in the coherent kernel instead of queueing them. */
  readonly inlineConeBoundaries?: boolean;
  /** Replace the one-thread queue-reset dispatch with a Dawn/Metal blit clear. */
  readonly clearConeQueueWithBlit?: boolean;
  /** Keep guided-upsample accumulators in native f16 registers. */
  readonly halfPrecisionLighting?: boolean;
  /** Drop the invocation-private GI page cache and re-fetch the direct table. */
  readonly dropGiPageCache?: boolean;
  /** Halve the canonical traversal stack for bounded-depth occupancy experiments. */
  readonly shortTraversalStack?: boolean;
  /** Quarter the canonical traversal stack as an overflow/fallback probe. */
  readonly tinyTraversalStack?: boolean;
  /**
   * Retain the original one-DDA-per-proxy brick fragment as the exact control
   * for the conservative coverage/resolve production arm.
   */
  readonly rasterPrimaryDirect?: boolean;
  /**
   * Retain the one-fragment-per-covering-proxy scene-primitive raster as the
   * exact control for its conservative coverage/resolve arm.
   *
   * Deliberately separate from `rasterPrimaryDirect`, which switches the *brick*
   * arm at the same time: an image comparison that moves both cannot say which
   * one moved the pixels.
   */
  readonly scenePrimitiveDirect?: boolean;
  /**
   * Drop only the brick raster's frag_depth write, leaving the empty-brick
   * discard in place. Brick proxies are disjoint along any ray and a brick
   * with no hit still discards, so the interpolated proxy exit depth picks the
   * same winner as the hit depth does; only the depth buffer's stored value
   * changes. That makes this arm the one with a sound redesign behind it, so
   * its delta is the recoverable part of {@link rasterPrimaryHsrProbe}.
   */
  readonly rasterPrimaryNoFragmentDepth?: boolean;
  /**
   * Upper-bound probe for tile-based hidden-surface removal in the brick
   * raster. Writing frag_depth and calling discard both leave a fragment's
   * depth and coverage unknown until it has been shaded, so either one alone
   * forces this GPU to shade every overlapping brick proxy rather than keeping
   * only the winner. The arm drops both, which is the most HSR the hardware
   * could ever recover; the delta against the default bounds what any sound
   * redesign is worth. The image is wrong here — missed bricks still shade —
   * so this is a timing probe only, never a rendering mode.
   */
  readonly rasterPrimaryHsrProbe?: boolean;
  /**
   * The same upper-bound probe, for the *scene primitive* raster rather than
   * the brick raster.
   *
   * {@link rasterPrimaryHsrProbe} bounds hidden-surface removal on proxies that
   * are subsets of disjoint SVO leaf cells, where overlap along a ray is rare.
   * Authored primitive proxies are not disjoint and overlap heavily — on the
   * hero pond a pixel is covered by roughly two dozen of them, and 84 % of the
   * scene's kinds resolve their hit by a 48-iteration sphere trace rather than
   * a closed-form root. Writing frag_depth and calling discard both leave a
   * fragment's depth and coverage unknown until it has been shaded, so this
   * pass currently marches every covering proxy instead of keeping the winner.
   *
   * The arm drops both. The image is wrong here — occluded primitives still
   * shade and the stored depth is the proxy's rather than the surface's — so
   * this is a timing probe only, never a rendering mode. Its delta bounds what
   * a sound front-to-back resolve can recover.
   */
  readonly scenePrimitiveHsrProbe?: boolean;
  /**
   * Retain the unbounded `0 .. DRY_MISS` march interval in the scene-primitive
   * raster as the exact control for the box-bounded one. See
   * `dryScenePrimitiveMarchSpan` for why the bounded interval is the contract
   * the marched kinds were written against.
   */
  readonly scenePrimitiveUnboundedMarch?: boolean;
  /**
   * Reject a leaf brick at entry against the conservative slab its own producer
   * fitted — a Laine-Karras contour, stored in the 24 spare bits of the node
   * record the in-brick DDA already loads. See `lib/svo-brick-contour.ts`.
   *
   * A leaf exists because it holds *at least one* solid cell, so the DDA
   * routinely walks a whole chord of a legitimately occupied brick and finds
   * nothing; grazing terrain, the pond surface and the sparse canopy all do it.
   * The slab clamps `[entry, brickExit]` in the same place `bounds` already
   * clamps and returns a miss when the interval collapses — straight-line ALU
   * strictly outside the dependent loop, which is the whole design constraint
   * (`bounds`, outside the loop, measured -5.9%; `macro`, inside it, +33%).
   *
   * Conservative by construction, so it is image-exact rather than a quality
   * dial: a moved frame hash is a soundness bug in the fit, never something to
   * tune an epsilon against.
   */
  readonly brickContour?: boolean;
  /**
   * Additionally advance the DDA's start to where the ray enters the slab.
   *
   * Worth roughly twice what the rejection alone is worth and **not
   * image-exact**: the walk's reported `t` is an incrementally accumulated
   * boundary vector, so starting it at a different cell reaches the same face
   * one ULP apart. Conservative either way — nothing is skipped — but ~1 % of
   * pixels move by one f16 ULP, which is the same reason `bounds` moves the
   * image. Off unless a lane has decided it will accept that.
   */
  readonly brickContourEntryClamp?: boolean;
  /**
   * Which walks the slab is allowed to reject in. Both by default; the single
   * arms exist to bisect an image difference onto the primary or onto the
   * bounded visibility twin, which are the only two consumers.
   */
  readonly brickContourPrimaryOnly?: boolean;
  readonly brickContourVisibilityOnly?: boolean;
  /**
   * The control that separates "this clamp deleted something" from "this clamp
   * changed the compiler's mind".
   *
   * Emits the whole decode and interval test and then consumes the result in a
   * branch that can never be taken (`accepted == 2.0`, when the function only
   * ever returns 0 or 1). Every value and every control-flow edge that the real
   * arm adds is present; nothing it computes can reach the frame. An image that
   * still moves under this probe moved because the shader around it was
   * recompiled, not because a brick was rejected.
   */
  readonly brickContourInertProbe?: boolean;
  /**
   * Which of the two expressions the clamped exit reaches. `escape` clamps only
   * the loop's escape test, `cell-exit` only the owner-run interval bound,
   * `both` (the default) does what a plain clamp does.
   */
  readonly brickContourExitScope?: "both" | "escape" | "cell-exit" | "cell-exit-inert";
  /**
   * Shade every opaque surface as neutral white — a clay render.
   *
   * Not a shortcut and not a stand-in for unauthored materials: while geometry
   * and lattice resolution are the work, albedo is the one channel that
   * *competes* with the thing being judged. A colour difference and a form
   * difference land in the same pixel, and the eye attributes both to whichever
   * it noticed first. Neutralising albedo leaves shape, shadow, occlusion and
   * silhouette carrying the whole image, which is what the reference plate is
   * being compared on right now.
   *
   * Material *resolution* is untouched — the id lookup, the publication check
   * and `dryInvalidSurfaceMaterial` all still run, and an unpublished or
   * malformed record still reports as invalid. Only the appearance is replaced.
   * Returning white before those checks would mean a scene whose material arena
   * never published rendered perfectly, which is precisely the class of failure
   * that hid the voxel path behind analytic shading for this whole program.
   *
   * Solidity is unaffected: a cell is solid when `identity & 0xffff` is nonzero,
   * so the material channel is still load-bearing occupancy data no matter what
   * is done with its colour.
   */
  readonly neutralSurfaceAlbedo?: boolean;
  /**
   * Re-derive an incoherent cone-prepass receiver by tracing the analytic scene
   * again, instead of taking it from the full-resolution primary G-buffer.
   *
   * The control for the G-buffer receiver, which is otherwise the default. See
   * `dryPrepassCoherentMain`: the analytic arm is `O(authored records)` per
   * boundary texel and is what made the cone prepass the one lighting stage that
   * scaled with scene size.
   */
  readonly analyticConeBoundaries?: boolean;
  /**
   * Compile the deferred kernel with the indirect gather ABSENT rather than
   * uniform-gated. `dryGlobalIllumination` becomes the exact value the
   * flag-off path returns — radiance 0, visibility 1, valid — so the image is
   * unchanged while the cone gather, its page caches and the tetra sampling
   * become dead code Dawn eliminates. This is the feature-specialised variant
   * the split-bundle cache keys on when global illumination is disabled; a
   * GI-capable kernel with the uniform off still pays for the never-taken
   * branch (see the unified-voxel purge: 13.25 → 8.96 ms, byte-identical).
   */
  readonly globalIlluminationAbsent?: boolean;
  /**
   * Run the analytic candidate-BVH walk of `traceStatic` first and unbounded,
   * as it historically did, instead of seeding it with the voxel-resolved hit.
   *
   * The control for the bounded order. Both produce the same surface; the
   * unbounded one descends every candidate box the ray meets anywhere along its
   * length, which is where the primary's record-count scaling lived.
   */
  readonly unboundedAnalyticPrimary?: boolean;
  /**
   * Count, per primary pixel, how many leaf bricks the megakernel actually
   * entered and how that ray terminated.
   *
   * `primaryLeafVisits` caps the per-pixel leaf loop, and both exhaustion arms
   * — the cap itself and `SVO_STATUS_WORK_EXHAUSTED`/`SVO_STATUS_STACK_OVERFLOW`
   * out of the cursor — were empty statement blocks that no lane counted. A ray
   * does not stop at the first brick it *touches*, it stops at the first solid
   * voxel, so the visit count is a distribution and the cap is only its ceiling;
   * without this there is no way to tell a converged frame from one silently
   * dropping its tail rays.
   *
   * Off is the shipping path and emits no WGSL at all: every site below is a
   * template hole that collapses to the empty string, the storage binding is
   * absent from the layout, and the buffer is never allocated. On, it costs one
   * histogram bucket plus four counter increments of atomic traffic per pixel,
   * which is a diagnostic price and not a rendering one.
   */
  readonly primaryLeafVisitHistogram?: boolean;
  /**
   * Publish exact per-pixel primary work counters into an rgba32uint texture.
   * Benchmark diagnostic only: the extra storage writes perturb the pass and
   * must never be used as the timed production arm.
   */
  readonly primaryWorkMap?: boolean;
  /**
   * Seed the primary megakernel with a rasterized conservative entry depth
   * instead of starting every ray at the root AABB the camera sits inside.
   *
   * Default on. `false` withdraws the whole pass — the compute cull, the depth
   * draw, the texture, and the seed binding and every consumer of it in the
   * fragment — so the arm it selects is exactly the pre-prepass program rather
   * than the same program with a never-taken branch left priced into it.
   *
   * The pass is the spatial half of the answer the retired temporal cache used
   * to give (docs/svo-primary-visibility-handoff.md part 3): current-frame
   * work, no camera coherence key, nothing carried across frames. What it
   * removes is the descent, not the visibility.
   *
   * The seed is also the cursor's near bound, not only an early-out test.
   * Moving `ray.tMin` off zero re-associates every interval the parametric
   * traversal computes from it, so a leaf's reported `tEnter` — and with it the
   * in-leaf DDA's first cell — can shift by an ULP. That is why the acceptance
   * oracle for this arm is the packed-surface and identity-media planes, which
   * carry visibility and identity and are bit-identical either way, rather than
   * the hardware-depth and image planes, which move at ULP scale on
   * `garden-svo-lighting`. Skipping the descent is the point of the pass, so
   * the drift is the price, not a defect.
   */
  readonly primaryEntryPrepass?: boolean;
  /**
   * Fetch octree records at record width instead of one scalar at a time, and
   * stop refetching the ones the cursor already holds.
   *
   * The structural arena is one buffer typed `array<u32>`, which makes every
   * node fetch eight separately range-clamped scalar loads and every leaf fetch
   * four — the shape the reference `array<SvoNode>` binding never had. This arm
   * types it `array<vec4u>` (two loads a node, one a leaf) and additionally
   * collapses the leaf path's duplicates: a primary leaf visit used to read the
   * same 32-byte node three times — cursor, `dryPrimaryLeafResolve`,
   * `svoNodeBounds` inside the payload walk — and the leaf record twice, and the
   * cursor now hands both out with the hit.
   *
   * **Off by default, because it does not pay.** Measured on Dawn/Metal at
   * 2488x1256 with the entry prepass on, frame median against the reference
   * fetch shape: `large-power-dam-break` 21.234 -> 21.234 ms, `garden-
   * svo-lighting` 24.183 -> 23.921 ms. Both are inside the lane's own ~1-3%
   * spread and the two scenes do not agree in sign. Bit-exact — all four
   * fingerprints match the reference arm on both scenes — so what it buys is
   * fewer loads and no measurable time, which says the primary is not bound on
   * structural fetch once the prepass has removed the descent from most pixels.
   */
  readonly traversalVectorRecords?: boolean;
  /**
   * Drop the arithmetic the parametric child expansion repeats per node.
   *
   * The segment octant is derived once per segment instead of three times, the
   * midpoint crossing multiplies by the reciprocal the cursor already holds
   * instead of dividing, and the Morton decode stops at the low address word
   * below level 11. Only the reciprocal is inexact, and only at the ULP that
   * decides whether a crossing is an exact tie; a tie routes the node through
   * the AABB fallback, which is a different expansion of the same node, not a
   * different answer.
   *
   * **Off by default, for the same reason as the fetch shape above.** Same lane:
   * `large-power-dam-break` 21.234 -> 20.972 ms, `garden-svo-lighting`
   * 24.183 -> 24.379 ms. Inside noise, opposite signs, and the two arms together
   * are 21.692 / 25.362 — no better than either alone. The parametric expansion
   * is not what the primary window is made of.
   */
  readonly traversalLeanExpansion?: boolean;
}


/** Reduced-rate cone-lighting prepass target contract. */
export const SVO_DRY_CONE_PREPASS_CONTRACT = Object.freeze({
  /** rg32uint packs 8-bit AO and eight 7-bit light visibilities exactly into 64 bits. */
  visibilityFormat: "rg32uint" as GPUTextureFormat,
  visibilityTargetCount: 1,
  /** rgba16float packing [distance, oct-normal xy, 11-bit feature/field/motion metadata]. */
  geometryFormat: "rgba16float" as GPUTextureFormat,
  /** Exact uint16 material + uint16 owner identity used by the isolated shading pass. */
  identityFormat: "r32uint" as GPUTextureFormat,
  /** HDR opaque radiance evaluated by a separate reduced-rate shading pass. */
  radianceFormat: "rgba16float" as GPUTextureFormat,
  /** Full-resolution visibility written only for compacted silhouette pixels. */
  silhouetteRefinementFormat: "rg32uint" as GPUTextureFormat,
  /** Core-WebGPU storage-capable state plane: 0 untouched, 1 valid, 2 failed. */
  silhouetteRefinementStateFormat: "r32uint" as GPUTextureFormat,
  /** Every user-shadable light slot is cached by the reduced-rate prepass. */
  maximumPrepassLights: SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
  /** Guided-upsample weight below this threshold publishes an explicit reconstruction failure. */
  minimumReconstructionWeight: 0.05,
} as const);


/** Indirect worklist/readback ABI for full-resolution silhouette refinement. */
export const SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT = Object.freeze({
  queueHeaderWords: 4,
  queuedWord: 0,
  exhaustedWord: 1,
  exactInvalidWord: 2,
  failureTotalWord: 3,
  /** Compatibility alias: the two-counter seam reports all failed refinements. */
  invalidWord: 3,
  indirectOffsetBytes: 0,
  counterWords: 2,
  counterSizeBytes: 2 * Uint32Array.BYTES_PER_ELEMENT,
  diagnosticCounterWords: 4,
  diagnosticCounterSizeBytes: 4 * Uint32Array.BYTES_PER_ELEMENT,
  workgroupSize: 64,
} as const);


/** `[exhausted, publication, sceneLimit, traversal, traceContract]` diagnostics. */
export const SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT = Object.freeze({
  exhaustedWord: 0,
  publicationWord: 1,
  sceneLimitWord: 2,
  traversalWord: 3,
  traceContractWord: 4,
  words: 5,
  sizeBytes: 5 * Uint32Array.BYTES_PER_ELEMENT,
} as const);


/** GPU-visible reasons a requested live-derived lighting sample failed closed. */
export const SVO_DRY_DERIVED_FAILURE = Object.freeze({
  ambientOcclusionPage: 1 << 0,
  directVisibilityPage: 1 << 1,
  globalIlluminationPage: 1 << 2,
  reducedReconstruction: 1 << 3,
} as const);


/** Persistent camera-independent cache used by the reduced split GI pass. */
export const SVO_DRY_WORLD_GI_CACHE_CONTRACT = Object.freeze({
  entryCount: 1 << 18,
  entryBytes: 16,
  probeBytes: 8,
  payloadBytes: 8,
  probeCount: 4,
  allocatedBytes: (1 << 18) * 16,
  frameBytes: 144,
  dynamicInfluenceCells: 12,
  dynamicInfluenceBodyRadii: 3,
} as const);


/** Phase-1 static directional-light cache. One bounded queue drains cold demand over several frames. */
export const SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT = Object.freeze({
  format: "rg32uint" as GPUTextureFormat,
  populationBudget: 16_384,
  queueHeaderWords: 8,
  queueEntryWords: 2,
  voxelsPerPage: SVO_NODE_MIP_LAYOUT.interiorSize ** 3,
  requestWordsPerPage: SVO_NODE_MIP_LAYOUT.interiorSize ** 3 / 32,
  bytesPerTexel: 8,
} as const);


/**
 * Composes the production dry fragment shader. Scale 1 must return the exact
 * historical string byte-for-byte: the bit-exact frame fingerprint gates on it,
 * and every insertion below is the empty string in that configuration.
 * Reduced scales add isolated geometry (`dryPrepassGeometryMain`), cone
 * visibility (`dryPrepassVisibilityMain`), and opaque-shading
 * (`dryPrepassShadeMain`) entries plus a depth/normal/identity-guided consumer.
 * Keeping primary traversal, cone marching, and shading in separate entries
 * prevents their register lifetimes from collapsing Metal occupancy; every
 * phase is encoded afresh each frame, including for dynamic scenes.
 */
export const SVO_DRY_SCENE_PIXEL_PROBE_GROUP = 1;


/**
 * Constants the pixel-trace probe mirrors, passed instead of imported so the
 * probe module stays free of this one.
 *
 * `primaryMode` is not a constant: it decides whether the probe instruments a
 * hierarchy walk at all, and instrumenting one the frame did not perform is the
 * defect this argument exists to prevent.
 */
export function svoDryScenePixelProbeOptions(
  primaryMode: SvoPixelTracePrimaryMode = "traced",
  brick: Pick<SvoPixelTraceProbeOptions,
    "brickOccupancyMode" | "brickContour" | "brickContourEntryClamp" | "brickContourExitScope"> = {},
): SvoPixelTraceProbeOptions {
  return {
    primaryMode,
    ...brick,
    group: SVO_DRY_SCENE_PIXEL_PROBE_GROUP,
    coneLodBlendBandWidth: SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH,
    maximumShadedLights: SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
    areaLightSamples: SVO_DRY_SCENE_AREA_LIGHT_SAMPLES,
    stableOcclusionConeSamples: SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES,
    primaryLeafVisitHardLimit: SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
    visibilityFlags: SVO_DRY_VISIBILITY_FLAGS,
  };
}


export function createSvoDrySceneFragmentWGSL(
  coneLightingScale: SvoConeLightingScale = 1,
  traversalMode: SvoDryTraversalMode = "hybrid",
  // `bounds` by default: the DDA walk is clamped to the occupied sub-box the
  // occupancy word already publishes. Measured -16.7% on a clean interleaved
  // lane, and conservative by construction — verified over the full population
  // (0 escapes across all 103,285 published leaf bricks). It composes with
  // `brickContour` for ~-24.7% together. `macro` remains a +33% regression and
  // `macro-hdda` is untouched; see `brickContourEntryClamp` for why the frame is
  // not byte-identical and why that is not an oracle worth holding this to.
  brickOccupancyMode: SvoBrickOccupancyMode = "bounds",
  shadingPath: SvoDryShadingPath = "inline",
  screenSpaceTerminationPixels = 0,
  /**
   * Appends the live pixel-trace probe entry point. Off by every production
   * call site, which is what keeps scale-1 output byte-identical to the string
   * the frame fingerprint gates on.
  */
  pixelProbe = false,
  /** Split-only path: pane discovery is rasterized over projected coverage. */
  rasterGlassDiscovery = false,
  /** Split-only path: moving rigid primary hits are rasterized from live BodyGPU records. */
  rasterRigidDiscovery = false,
  /** Reduced split-only experiment: execute one deterministic cone per compute lane. */
  coneFanout = false,
  experiments: SvoDryOptimizationExperiments = {},
): string {
  if (pixelProbe && (coneLightingScale !== 1 || shadingPath !== "inline")) {
    throw new RangeError("The pixel-trace probe requires the inline, full-rate dry-scene composition");
  }
  if (rasterGlassDiscovery && shadingPath !== "split") {
    throw new RangeError("Raster glass discovery requires split shading");
  }
  if (rasterRigidDiscovery && shadingPath !== "split") {
    throw new RangeError("Raster rigid discovery requires split shading");
  }
  if (coneFanout && (shadingPath !== "split" || coneLightingScale === 1)) {
    throw new RangeError("Cone fan-out requires reduced split shading");
  }
  if (!SVO_DRY_TRAVERSAL_MODES.includes(traversalMode)) {
    throw new RangeError(`Unsupported dry-scene traversal mode: ${traversalMode}`);
  }
  // The mode only replaces the split path's primary entry. Inline variants of
  // the same module — the reduced cone prepass, the diagnostic megakernel —
  // legitimately keep traversing, so they simply omit the raster entries.
  const rasterPrimary = traversalMode === "raster-primary" && shadingPath === "split";
  // The rasterized conservative entry depth is the split megakernel's seed and
  // nothing else's. `raster-primary` already resolves visibility from proxy
  // boxes of its own and has no root descent left to shorten, and the inline
  // composition has no pass ordered before its fragment to write a seed from.
  const primaryEntrySeed = shadingPath === "split" && traversalMode !== "raster-primary"
    && experiments.primaryEntryPrepass !== false;
  // The point of the mode is to unfuse the megakernel: panes reach the brick
  // fragment only as an already-rasterized key, never as a loop. Bodies are
  // likewise a renderer-level requirement, checked where the passes are wired.
  if (rasterPrimary && !rasterGlassDiscovery) {
    throw new RangeError("Raster-primary traversal requires raster glass discovery");
  }
  if (brickOccupancyMode !== "off" && brickOccupancyMode !== "bounds"
    && brickOccupancyMode !== "macro" && brickOccupancyMode !== "macro-hdda") {
    throw new RangeError(`Unsupported dry-scene brick occupancy mode: ${brickOccupancyMode}`);
  }
  if (shadingPath !== "inline" && shadingPath !== "split") {
    throw new RangeError(`Unsupported dry-scene shading path: ${shadingPath}`);
  }
  if (!Number.isFinite(screenSpaceTerminationPixels) || screenSpaceTerminationPixels < 0) {
    throw new RangeError("Dry-scene screen-space termination must be a non-negative finite pixel count");
  }
  const screenSpaceTerminationSupported = (traversalMode === "canonical" && shadingPath === "inline")
    || (traversalMode === "raster-primary" && shadingPath === "split");
  if (screenSpaceTerminationPixels > 0 && !screenSpaceTerminationSupported) {
    throw new RangeError("Screen-space termination requires canonical inline or raster-primary split traversal");
  }
  const reduced = coneLightingScale !== 1;
  const split = shadingPath === "split";
  const splitGroup = reduced ? 2 : 1;
  // One renderer also compiles inline cone-prepass modules from this builder;
  // the diagnostic belongs only to its traced split visibility module.
  const primaryWorkMap = experiments.primaryWorkMap === true && split && !rasterPrimary;
  // Compile-time because the codec is: producer and consumer must agree about a
  // shift, and the selector is the same pure function of the environment that
  // the voxeliser resolves its own arm from. A mismatch would render a wrong
  // scene rather than fail, which is exactly what emitting the shared codec —
  // instead of hand-matching the bits here — is for.
  const sceneGeometryFormat = octreeLiveSceneSceneGeometryFormat();
  // Compile-time for the same reason the geometry codec is, and resolved from the
  // same pure function of the environment the world itself resolves its arm from.
  // The lane *addresses* still arrive by uniform (see `payloadLaneWordOffset`):
  // which decode to compile is a property of the world's layout, where the
  // addresses are a property of one arena, and a renderer outlives arenas.
  const leafPayloadMode = octreeLiveSceneLeafPayloadMode();
  // The banded record width is not a free axis — `resolveSparseBrickPayloadLayout`
  // refuses the pairing outright — so a shader compiled for it would be a decoder
  // for an arena that cannot exist. Fail here rather than emit one.
  if (leafPayloadMode === "banded" && sceneGeometryFormat !== "f16-unorm8") {
    throw new RangeError("The banded leaf payload requires the f16-unorm8 record width");
  }
  // Identity, in the two shapes a marcher needs: `sceneIdentityAt` for the sites
  // that resolve one cell, and `sceneIdentitySourceAt`/`sceneIdentityOf` for the
  // DDAs, which hoist everything that depends only on the leaf out of the loop.
  //
  // `records: false` — smooth primary reconstruction reads the retained dense
  // geometry lane. Compact records cannot cover every first-hit voxel at leaf
  // boundaries, so they remain outside this identity-only codec.
  const sceneIdentityWGSL = /* wgsl */ `${leafPayloadMode === "dense" ? "" : sparseBrickBandedLeafCodecWGSL({
    occupancyBase: "dry.payloadLanes.x", recordMaskBase: "dry.payloadLanes.y",
    headerBase: "dry.payloadLanes.z", blobBase: "dry.payloadLanes.w",
    recordsBase: "0u",
    load: (index) => `scenePayload[${index}]`, mode: leafPayloadMode, records: false,
  })}
${sparseBrickSceneIdentityCodecWGSL({
    mode: leafPayloadMode, materialOwnerBase: "dry.payloadLanes1.y",
    load: (index) => `scenePayload[${index}]`,
  })}
${sparseBrickSceneIdentityWordCodecWGSL()}
// The bound every identity read tests against.
//
// It replaces \`arrayLength(&materialOwners)\`, which stopped being the lane's
// length the moment binding 3 became the whole arena — and which a banded world
// has no lane to measure at all. Published rather than derived so the dense and
// banded arms reject exactly the same voxel indices.
fn dryVoxelCapacity()->u32{return dry.payloadLanes1.z;}`;
  // Coverage and the baked normal are the two halves of the sub-voxel surface
  // sample. The normal supplies orientation; coverage inverts the voxelizer's
  // planar coverage law to recover the plane's signed offset from cell centre.
  //
  // The dense geometry lane is deliberately used for every payload mode. A
  // banded world still retains that lane as the encoder's staging input today,
  // and first-hit voxels are not guaranteed to own compact records at a leaf
  // boundary. Reading a record opportunistically would therefore make the
  // reconstruction acquire seams at exactly those boundaries.
  const sceneSurfaceGeometryWGSL = /* wgsl */ `
${sparseBrickSceneGeometryCodecWGSL("f16-unorm8")}
fn drySceneFractionOfVoxel(voxel:u32)->f32{
  let descriptor=dry.payloadLanes1.w;
  let stride=(descriptor>>8u)&0xffu;
  let fractionWord=(descriptor>>16u)&0xffu;
  let packed=(descriptor&(1u<<24u))!=0u;
  let word=dry.payloadLanes1.x+voxel*stride+fractionWord;
  if(word>=arrayLength(&scenePayload)){return 0.0;}
  let fraction=select(bitcast<f32>(scenePayload[word]),sceneFractionOf(scenePayload[word],voxel),packed);
  return clamp(fraction,0.0,1.0);
}`;
  /**
   * The per-leaf identity storage, resolved once before a voxel scan.
   *
   * Emitted **only on the `dense` arm**. `dense` needs the lane base for every
   * cell because its solidity test *is* an identity load. The two mask arms need
   * nothing per cell but the occupancy bit, so a hoisted source there is spent on
   * every leaf the ray crosses — including the ones it misses — to save nothing on
   * the at most one cell per ray that hits.
   *
   * Measured at **0.17 pp** on `hero-garden-hose` at depth 1, which is inside that
   * lane's noise. It is here because it is the honest structure, not because it
   * paid: the residual `banded` cost is the identity indirection itself, three
   * arena regions against `dense`'s one flat word. See
   * {@link sparseBrickSceneIdentityCodecWGSL}.
   */
  const cellIdentitySourceWGSL = (voxel: string): string =>
    leafPayloadMode === "dense" ? `let identitySource=sceneIdentitySourceAt(${voxel});` : "";
  /**
   * One cell of a voxel scan: reject it, or resolve its identity.
   *
   * **This is where the 1-bit occupancy mask is spent.** The `dense` arm has no
   * mask, so solidity is the low half of a 4-byte identity word and every
   * *rejected* cell pays a strided 4-byte load — that is the walk's hottest load
   * and the stride asymmetry the mask exists to kill. Under `occupancy` and
   * `banded` the question is one bit of a word shared with 31 neighbours, and the
   * identity — header, palette, prefix popcount, normal — is resolved only for the
   * cell that answers yes, which is at most one per ray because the walk returns
   * on it.
   *
   * The `dense` arm keeps the shipped expression — the same two statements in the
   * same order, differing from what it replaced only in line breaks — so the arm
   * every hash baseline was recorded against is not perturbed by this. Verified on
   * device: `dense`, `occupancy` and `banded` all render `hero-garden-hose` to
   * `0x8553a29b`.
   *
   * `onSolid` runs with `identity` in scope.
   */
  const cellSolidGateWGSL = (index: string, onSolid: string): string =>
    leafPayloadMode === "dense"
      ? `let identity=sceneIdentityOf(identitySource,${index});if(sceneIdentitySolid(identity)){${onSolid}}`
      : `if(sceneIdentitySolidAt(${index})){let identity=sceneIdentityAt(${index});${onSolid}}`;
  const fastDeferred = split && experiments.opaqueDirectionalCones === true;
  const voxelLightCache = split && experiments.voxelLightCache !== false;
  const edgeReceiverRecovery = reduced && experiments.edgeReceiverRecovery !== false;
  // Full-rate/inline shaders still own the diagnostic overlay. The reduced
  // split shaders are never selected while that overlay is active, so their
  // counters are dead production state and can be removed safely.
  if (experiments.halfPrecisionLighting && !reduced) {
    throw new RangeError("Half-precision lighting is restricted to reduced-rate shaders");
  }
  // What a prepass texel whose 2x2 full-resolution neighbourhood disagrees does.
  //
  // Both analytic arms re-derive the receiver by tracing the scene again from
  // the prepass ray, and that trace walks the candidate BVH with an exact
  // primitive intersection per leaf — `O(authored records)` per boundary texel,
  // on roughly half of them. Measured on the hero: 43 137 of 92 000 texels at
  // 501 records and 52 497 at 5 039, with the cost *per* re-trace growing 4.05x
  // for 10.06x records. The queue depth barely moves; the walk is the scaling.
  //
  // The default takes the receiver from the full-resolution primary G-buffer
  // instead, which already holds the exact primary hit for every one of those
  // four samples — the same plane `drySilhouetteRefineMain` consumes rather than
  // retraces. It is a quality decision and not a free one: no full-resolution
  // pixel centre coincides with the prepass ray, which is why the homogeneity
  // test exists at all, so the nearest of the four is chosen rather than a
  // fixed corner. At a silhouette that is the foreground surface, which is the
  // receiver a cone from this texel would actually have found, and it is the
  // one the reduced-rate reconstruction can recover the other side of.
  //
  // Measured on the hero at 800x460, cone scale 0.5, all arms interleaved in one
  // process (serialized submit-to-fence): 292.5 -> 227.4 ms at 501 records and
  // 1564.4 -> 1190.7 ms at 5 039, with the boundary queue reading zero and the
  // depth plane unchanged against the analytic reference at both rungs.
  const inlineBoundaryWGSL = experiments.inlineConeBoundaries
    ? "let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(coordinate,opaque,ray[0],ray[1]);return;"
    : experiments.analyticConeBoundaries
      ? "let queueIndex=atomicAdd(&dryPrepassBoundaryQueue.count,1u);dryPrepassBoundaryQueue.coordinates[queueIndex]=globalId.y*dimensions.x+globalId.x;return;"
      : "var nearest=3u;var nearestDepth=DRY_MISS;for(var sample=0u;sample<4u;sample+=1u){let candidate=primaryGeometry[sample].w;if(candidate<nearestDepth){nearestDepth=candidate;nearest=sample;}}referenceGeometry=primaryGeometry[nearest];referenceIdentity=primaryIdentity[nearest];";
  // Secondary rays keep the measured production traversal; only the primary
  // changes shape in raster-primary mode.
  //
  // Keyed on the traversal mode alone, not on whether *this* composition emits
  // the raster entries. The inline compositions of this module — the diagnostic
  // megakernel, the pixel probe — do not rasterize, but they run beside a frame
  // that does, and their secondaries must be the traversal that frame's
  // secondaries use. Gating this on the split path instead left "raster-primary"
  // matching neither the canonical nor the compact branch below, so those
  // compositions silently fell through to the wide-fanout cursor.
  const secondaryTraversalMode = traversalMode === "raster-primary" ? "canonical-parametric" : traversalMode;
  const hsrProbe = experiments.rasterPrimaryHsrProbe === true;
  const scenePrimitiveHsrProbe = experiments.scenePrimitiveHsrProbe === true;
  const neutralSurfaceAlbedo = experiments.neutralSurfaceAlbedo === true;
  // The per-brick conservative slab, on by default. See `brickContour`.
  //
  // Measured 2026-08-07 on a clean interleaved lane (12 pairs, scene fingerprint
  // checked before and after every run): **-19.2%** with the entry clamp, and
  // -24.7% combined with `bounds`. The slab is conservative by construction and
  // that was verified over the whole population, not a sample — every solid cell
  // of all 103,285 published leaf bricks against its stored summary, **0 escapes,
  // worst overshoot 0.0000 cells**. No hit can be deleted by it.
  //
  // The surprise worth recording: **outright rejection is worth ~0%.** An
  // exit-only arm, which still rejects every chord the slab misses, measures
  // noise (-0.99, -0.45, -0.29, +0.13%). All of the win is `brickContourEntryClamp`
  // advancing past leading empty cells in bricks the ray *does* hit. The census
  // says 27.0% of chords are rejected outright and survivors shorten by 27.8%;
  // only the second number buys anything. Chord shortening is what predicts
  // whether this generalises to another scene, not rejection rate.
  const brickContour = experiments.brickContour !== false;
  const analyticPrimaryUnbounded = experiments.unboundedAnalyticPrimary === true;
  // The first solid cell along the DDA bounds the surface. Voxel-flat mode keeps
  // the face at `entry`; smooth mode intersects the sub-voxel tangent plane
  // encoded by that cell's coverage and baked normal. The plane must intersect
  // its own cell or it falls back to the face, so reconstruction cannot create a
  // visibility hole.
  //
  // The owner is gone from the returned hit. A voxel has no back-pointer to an
  // authored record any more, so hover, picking and per-owner suppression of
  // voxel surfaces are gone with it; analytic hits — rigid bodies, glass — still
  // carry their own owner and still suppress.
  const primaryVoxelSurfaceWGSL = /* wgsl */ `if(cellSolid){
      let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);
      let faceNormal=dryVoxelFaceNormal(cellBounds,ro+rd*entry);
      let shaded=dryShadingNormal(cellIdentity,faceNormal);
      let cellExit=min(nextT.x,min(nextT.y,nextT.z));
      let surfaceT=drySmoothVoxelSurfaceT(cellIdentity,payloadIndex,cellBounds,ro,rd,entry,cellExit,shaded.normal);
      return DryHit(surfaceT,shaded.normal,sceneIdentityMaterial(cellIdentity),DRY_OWNER_NONE,
        shaded.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
    }`;
  // The exact hit is unchanged either way — this only decides how much empty
  // interval the sphere trace is asked to walk before it reaches the solid.
  const scenePrimitiveMarchSpanWGSL = experiments.scenePrimitiveUnboundedMarch === true
    ? "let exact=primitiveHit(record,ro,rd,0.0,DRY_MISS);"
    : "let span=dryScenePrimitiveMarchSpan(record,ro,rd);let exact=primitiveHit(record,ro,rd,span.x,span.y);";
  if (scenePrimitiveHsrProbe && traversalMode !== "raster-primary") {
    throw new RangeError("The scene primitive raster depth experiment only applies to raster-primary traversal");
  }
  const noFragmentDepth = hsrProbe || experiments.rasterPrimaryNoFragmentDepth === true;
  const directRasterPrimary = experiments.rasterPrimaryDirect === true || noFragmentDepth;
  if (directRasterPrimary && traversalMode !== "raster-primary") {
    throw new RangeError("The brick raster depth experiments only apply to raster-primary traversal");
  }
  const canonicalTraversal = secondaryTraversalMode === "canonical" || secondaryTraversalMode === "canonical-parametric";
  const wideTraversalWGSL = canonicalTraversal || secondaryTraversalMode === "compact"
    ? ""
    : createWebgpuSvoWideFanoutTraversalWGSL({ arena: {
      binding: 5,
      pageOffset: "dry.derivedTraversal.x",
      descriptorOffset: "dry.derivedTraversal.y",
    } });
  const compactTraversalWGSL = secondaryTraversalMode === "compact" ? createWebgpuSvoCompactTraversalWGSL(5) : "";
  const compactTraversal = secondaryTraversalMode === "compact";
  const traversalVectorRecords = experiments.traversalVectorRecords === true;
  // Only the canonical cursor holds a 32-byte node record to hand over, so the
  // primary leaf path takes its records from the visit on that arm alone; the
  // compact and wide arms keep fetching their own, which is what they have.
  const primaryVisitRecords = traversalVectorRecords && canonicalTraversal;
  const canonicalTraversalWGSL = createWebgpuSvoTraversalWGSL({ arena: {
      binding: 2,
      controlOffset: "dry.structureOffsets.x",
      nodeOffset: "dry.structureOffsets.z",
      leafOffset: "dry.structureOffsets.w",
      vectorRecords: traversalVectorRecords,
    },
    childEnumeration: secondaryTraversalMode === "canonical-parametric" ? "parametric" : "aabb",
    publishVisitRecords: primaryVisitRecords,
    leanExpansion: experiments.traversalLeanExpansion === true,
    stackCapacity: experiments.tinyTraversalStack ? 8 : experiments.shortTraversalStack ? 16 : 32 });
  const screenSpaceTraversalWGSL = screenSpaceTerminationPixels > 0
    ? createSvoScreenSpaceTraversalWGSL(canonicalTraversalWGSL) : "";
  const lodDescentWGSL = screenSpaceTerminationPixels > 0 ? svoLodDescentWGSL : "";
  const leafAccessWGSL = compactTraversal ? /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoCompactNodeBounds(svoCompactNodes[nodeIndex],dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).links.w;}
fn dryLeafLevel(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).address.z;}
` : /* wgsl */ `
fn dryLeafBounds(nodeIndex:u32)->mat2x3f{return svoNodeBounds(svoNodeLoad(nodeIndex),dry.mapping);}
fn dryLeafFlags(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).links.w;}
fn dryLeafLevel(nodeIndex:u32)->u32{return svoNodeLoad(nodeIndex).address.z;}
`;
  const liveLeafLifecycleWGSL = /* wgsl */ `
fn dryLeafStructuralPlanar(hit:SvoTraversalHit)->bool{
  if(hit.nodeIndex>=svoControlLoad(0u)){return false;}
  let node=svoNodeLoad(hit.nodeIndex);let leafIndex=node.links.z;
  if(leafIndex==SVO_INVALID||leafIndex>=svoControlLoad(1u)){return false;}
  let leaf=svoLeafLoad(leafIndex);
  return leaf.topology.x==hit.nodeIndex
    &&leaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY;
}
fn dryLeafCurrent(hit:SvoTraversalHit)->bool{
  // A planar terminal is immutable structural topology and deliberately owns
  // no voxel payload for the scene voxeliser to stamp current. Requiring the
  // payload lifecycle here skips the terminal before its exact intersection is
  // even attempted, so validate its node/leaf link instead. Ordinary leaves
  // retain the live publication gate unchanged.
  return dryLeafStructuralPlanar(hit)
    ||svoBrickLifecycleCurrent(svoBrickLifecycleDecode(dryLeafFlags(hit.nodeIndex)));
}
`;
  // The shading normal, read back out of the voxel that was hit.
  //
  // This used to be a three-tier normal ladder — the owner's analytic normal, a
  // heightfield gradient for the ground, and an eight-tap trilinear stencil of
  // the stored distance field underneath both — and all three normal paths are
  // gone. The
  // voxeliser now evaluates the winning primitive's own outward normal at bake
  // time and packs it oct8 into the high half of the identity word the DDA
  // already loaded (`sparseBrickSceneIdentityWordCodecWGSL`), so the shaded
  // direction costs an unpack of a register.
  //
  // The producer is the right place for it. It knows exactly which primitive
  // filled the cell, it is already holding that record, and it runs once per
  // voxel at init instead of once per pixel per frame. Everything the renderer
  // needed the analytic scene *for* — records, owner ids, the heightfield — was
  // needed only to answer this one question.
  //
  // What the ladder's floor was, `dryVoxelFaceNormal`, survives as the fallback
  // for a voxel whose bake found no gradient: six axis directions, which terrace,
  // and that is the correct thing to draw where the field genuinely has no
  // surface orientation. It is not reachable on any authored surface.
  const surfaceReconstructionWGSL = /* wgsl */ `
${sceneSurfaceGeometryWGSL}
struct DryShadingNormal{normal:vec3f,featureId:u32}
/**
 * Presentation-wide six-face classification.
 *
 * Most authored scenery reaches the frame through the payload DDA and already
 * owns an exact entered-cell face.  Solver rigid bodies are the exception: they
 * remain analytic so fluid can collide with them, and their curved normal could
 * win the primary depth test over the matching voxel scenery.  In voxel-flat
 * mode that leaked a smooth circular patch into an otherwise faceted object.
 * Classifying every final opaque normal to its dominant signed axis makes the
 * scene contract independent of which primary producer happened to win.
 */
fn dryVoxelFaceAxis(normalIn:vec3f)->vec3f{
  let normal=normalize(normalIn);let magnitude=abs(normal);var face=vec3f(0.0);
  if(magnitude.x>=magnitude.y&&magnitude.x>=magnitude.z){face.x=select(-1.0,1.0,normal.x>=0.0);}
  else if(magnitude.y>=magnitude.z){face.y=select(-1.0,1.0,normal.y>=0.0);}
  else{face.z=select(-1.0,1.0,normal.z>=0.0);}
  return face;
}
fn dryPresentationNormal(normal:vec3f)->vec3f{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u)!=0u){return dryVoxelFaceAxis(normal);}
  return normalize(normal);
}
fn dryPresentationHit(hitIn:DryHit)->DryHit{
  var hit=hitIn;
  if(hit.t<DRY_MISS){hit.normal=dryPresentationNormal(hit.normal);}
  return hit;
}
/**
 * A connected occupancy field has no internal boundary between two adjacent
 * cells, so face normals alone turn every coplanar run into one unbroken slab.
 * The references treat each occupied cell as an individually articulated cube.
 * Preserve the exact six-axis geometric normal, but add a narrow, antialiased
 * contact seam on the two in-plane lattice coordinates of the visible face.
 *
 * The analytic pixel footprint keeps the seam near one screen pixel and fades
 * it once a cell becomes sub-pixel. The lattice remains the actual finest-cell lattice; this does not
 * coarsen occupancy, alter silhouettes, or reopen the coping gaps caused by a
 * coarse geometry bake.
 */
fn dryVoxelFaceEdgeFactor(position:vec3f,faceNormal:vec3f,depth_m:f32,fieldSource:u32)->f32{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u)==0u){return 1.0;}
  // The seam is a presentation of individually articulated *voxel* faces. An
  // analytic terminal retains the exact thin surface specifically so a large
  // floor or wall reads as one plane; projecting the finest-cell lattice over
  // that hit would throw away the visual half of the planar cutover while the
  // traversal still paid for the exact answer.
  if(fieldSource==DRY_GBUFFER_FIELD_ANALYTIC){return 1.0;}
  let cellCoordinate=(position-dry.nodeMipOrigin.xyz)/max(dry.mapping.cellSize,vec3f(1e-6));
  let phase=abs(fract(cellCoordinate)-vec3f(.5));
  let edgeDistance=vec3f(.5)-phase;
  let worldPixel=max(2.0*depth_m*cameraTanHalfFov()/max(uniforms.viewport.y,1.0),1e-6);
  let footprint=max(vec3f(worldPixel)/max(dry.mapping.cellSize,vec3f(1e-6)),vec3f(1e-4));
  let tangentMask=vec3f(1.0)-abs(faceNormal);
  var edge=1.0;var maximumFootprint=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    if(tangentMask[axis]>.5){
      let width=max(.075,.55*footprint[axis]);
      edge=min(edge,smoothstep(0.0,width,edgeDistance[axis]));
      maximumFootprint=max(maximumFootprint,footprint[axis]);
    }
  }
  let resolved=1.0-smoothstep(.70,1.20,maximumFootprint);
  return mix(1.0,mix(.78,1.0,edge),resolved);
}
/**
 * The baked normal, or the entered face where none was baked.
 *
 * The right-angle test is retained from the analytic arm and for the same
 * reason: a normal more than ninety degrees from the face the ray entered
 * through is not this cell's surface however exactly it was computed. It can
 * happen legitimately — the bake samples the voxel *centre*, which is up to a
 * cell radius off the surface, so a thin feature crossing the cell can hand back
 * the far side's orientation — and the face is the better answer there.
 *
 * *More than* ninety degrees. It used to reject ninety exactly, and ninety
 * exactly is a flat surface entered through one of its side faces: dot(+y, -z)
 * is zero to the bit for a box's baked normal against an axis face. That is the
 * grazing view of a floor, and it is where the baked normal matters most —
 * rejecting it there paints a floor voxel with the orientation of the wall of
 * its own cube, which under a lamp pointing straight down is black. It is the
 * dark skirting seen through the glass at the base of a tank: the studio floor's
 * top row of voxels, entered edge-on through the wall, shading as if each were a
 * vertical face. From outside the glass that side is never entered and the same
 * voxel shades as floor, which is why it read as a glass bug.
 *
 * The feature id is always smooth. It was the analytic normal's own hard-feature
 * classification and there is nowhere in sixteen bits to keep it; the only thing
 * downstream of it for a voxel hit is the contact-visibility ray bias, 0.025
 * cells against 0.05.
 */
fn dryShadingNormal(identity:u32,faceNormal:vec3f)->DryShadingNormal{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u)!=0u){
    return DryShadingNormal(faceNormal,SVO_FEATURE_SMOOTH);
  }
  if(sceneIdentityHasNormal(identity)){
    let baked=sceneIdentityNormal(identity);
    if(dot(baked,faceNormal)>-1e-4){return DryShadingNormal(baked,SVO_FEATURE_SMOOTH);}
  }
  return DryShadingNormal(faceNormal,SVO_FEATURE_SMOOTH);
}
/**
 * Intersect the sub-voxel tangent plane represented by this cell.
 *
 * The voxelizer's coverage law is
 *
 *   fraction = 0.5 - signedOffset / (2 * cellRadius)
 *
 * so the stored fraction recovers a point on a plane whose orientation is the
 * baked normal. Unlike changing only \`DryHit.normal\`, this changes \`DryHit.t\`
 * and therefore the visible surface and hardware depth. The intersection must
 * remain inside the occupied cell; a full/binary cell or a plane that misses
 * its own cell has insufficient sub-voxel evidence and keeps the watertight
 * voxel face instead of opening a crack.
 */
fn drySmoothVoxelSurfaceT(identity:u32,voxel:u32,bounds:mat2x3f,ro:vec3f,rd:vec3f,
  entry:f32,cellExit:f32,normalIn:vec3f)->f32{
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u)!=0u
    ||!sceneIdentityHasNormal(identity)){return entry;}
  let fraction=drySceneFractionOfVoxel(voxel);
  if(!(fraction>0.0&&fraction<1.0)){return entry;}
  let normal=normalize(normalIn);
  let denominator=dot(rd,normal);
  if(abs(denominator)<1e-6){return entry;}
  let extent=bounds[1]-bounds[0];
  let centre=0.5*(bounds[0]+bounds[1]);
  let radius=0.5*length(extent);
  let signedOffset=radius*(1.0-2.0*fraction);
  let planePoint=centre-normal*signedOffset;
  let candidate=dot(planePoint-ro,normal)/denominator;
  let tolerance=max(1e-6,1e-4*radius);
  if(candidate<entry-tolerance||candidate>cellExit+tolerance){return entry;}
  let point=ro+rd*candidate;
  if(any(point<bounds[0]-vec3f(tolerance))||any(point>bounds[1]+vec3f(tolerance))){return entry;}
  return clamp(candidate,entry,cellExit);
}
`;
  // The compile flag decides whether the LOD machinery exists; the uniform
  // decides what it does. A build without it is the bit-exact reference image,
  // and a runtime threshold of zero reproduces that image from the LOD build —
  // which is what lets the panel drag the slider to zero and get the reference
  // rather than "very nearly" it.
  const lodUniformWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn dryLodMode()->u32{return u32(dry.lod.y);}
fn dryLodFixedLevel()->u32{return u32(dry.lod.z);}
/** Authored at the contract's reference height, rescaled here so it stays angular. */
fn dryLodThresholdPixels()->f32{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return 0.0;}
  return dry.lod.x*uniforms.viewport.y/${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels};
}
/** Cells per aggregate step for one brick. One is the exact per-cell walk. */
fn dryLodCellStride(brickBounds:mat2x3f,leafLevel:u32)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return svoLodCellStrideForLevel(leafLevel,dryLodFixedLevel(),brickSize);}
  return svoLodScreenSpaceCellStride(brickBounds,brickSize,uniforms.cameraPosition.xyz,
    uniforms.viewport.y,cameraTanHalfFov(),dryLodThresholdPixels());
}
` : "";
  const primaryTraversalCursorWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn drySvoShouldTerminateNodeScreenSpace(bounds:mat2x3f,level:u32)->bool{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return level>=dryLodFixedLevel();}
  return svoShouldTerminateNodeScreenSpace(bounds,uniforms.cameraPosition.xyz,uniforms.viewport.y,cameraTanHalfFov(),dryLodThresholdPixels(),level,0u);
}
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  return svoTraversalContinuationNextScreenSpace(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
` : /* wgsl */ `
fn dryTraversalCursorNextPrimary(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return dryTraversalCursorNext(ray,mapping,cursor);}
`;
  const screenSpaceProxyWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
const DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY:u32=15u;
const DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY:u32=14u;
fn dryScreenSpaceProxyHit(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit{
  let bounds=svoNodeBounds(svoNodeLoad(hit.nodeIndex),dry.mapping);let point=ro+rd*hit.tEnter;
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return DryHit(hit.tEnter,normal,0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(hit.level,0u,0u));
}
` : "";
  const screenSpaceProxyTraceWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(leaf.status==SVO_STATUS_SCREEN_SPACE_PROXY){return dryScreenSpaceProxyHit(ro,rd,leaf);}` : "";
  const screenSpaceProxyShadeWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `if(hit.fieldSource==DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY){let depthBand=clamp(f32(hit.aux.x)/21.0,0.0,1.0);return mix(vec3f(.02,.06,.18),vec3f(1.0,.04,.72),depthBand);}` : "";
  const screenSpacePrimaryProxyWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
fn dryPrimaryBoundsSubPixel(bounds:mat2x3f)->bool{
  return svoShouldTerminateNodeScreenSpace(bounds,uniforms.cameraPosition.xyz,uniforms.viewport.y,
    cameraTanHalfFov(),dryLodThresholdPixels(),0u,0u);
}
// Whole-brick collapse: everything this instance can draw, measured at once.
//
// It used to measure the brick's camera-nearest *cell* and collapse all eight
// cells across on that answer — an eight-fold over-coarsening that stayed
// invisible while 25 mm cells put its onset near 14 m, and became structural at
// 6.25 mm, where it starts around 3.5 m and so claims most of the garden.
// Compounding it, the bounds handed in are the occupied sub-AABB the emit
// kernel publishes (webgpu-svo-brick-raster.ts:727), not the brick box, so
// span/brickSize understated the cell by as much as another eight per axis on
// exactly the sparse foliage bricks where the error is most visible.
//
// The sub-AABB is the right thing to measure *as a whole*: it is the tight
// bound on every surface the brick can produce, so a sub-threshold sub-AABB
// genuinely cannot express more than one proxy. Intermediate granularity is now
// the aggregate stride's job (dryLodCellStride), not this predicate's.
fn dryPrimaryBrickProxySubPixel(bounds:mat2x3f,leafLevel:u32)->bool{
  if(dryLodMode()==SVO_LOD_MODE_FIXED_LEVEL){return dryLodFixedLevel()<=leafLevel;}
  return dryPrimaryBoundsSubPixel(bounds);
}
fn dryPrimaryProxyNormal(bounds:mat2x3f,point:vec3f)->vec3f{
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);
  normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return normal;
}
// The entry face, not the interval midpoint.
//
// The midpoint pushed the proxy half an aggregate behind the surface it stands
// for — 25 mm at a 50 mm brick — and this hit is depth-tested against the exact
// tier's, so the bias was not cosmetic: it let an exact neighbouring fragment
// win a silhouette pixel the proxy actually owned. The entry face is the
// nearest point of the aggregate the ray touches, which is what a solid
// aggregate presents, and it makes a stride-one proxy agree with the exact
// voxel hit exactly rather than approximately.
fn dryPrimaryVoxelProxyHit(ro:vec3f,rd:vec3f,bounds:mat2x3f,tEnter:f32,identity:u32)->DryHit{
  // The proxy keeps the aggregate's face rather than the baked normal: it stands
  // for a whole block of cells, and one cell's surface orientation is not that
  // block's.
  return DryHit(tEnter,dryPrimaryProxyNormal(bounds,ro+rd*tEnter),sceneIdentityMaterial(identity),
    DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,
    DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
}
// The same test traceLeafPayload applies, and it has to be: this function is the
// only resolve a pixel classified into the LOD tier ever gets.
fn dryLodCellSolid(identity:u32)->bool{return sceneIdentitySolid(identity);}
// One identity standing for the whole brick, for the ray that crosses it
// without crossing a solid cell.
//
// The macrocell mask makes this bounded and exact rather than a search: a set
// bit promises at least one solid cell inside that 4^3 block, so the first set
// bit's block is normally the only one scanned, and a typical surface brick
// answers within a few fetches of entering it. A block whose cells all belong
// to a suppressed owner is the one case that falls through to the next bit,
// and a brick that is suppressed outright returns zero — which is a genuine
// absence of surface, not a missing answer.
fn dryLodBrickRepresentativeIdentity(voxelOffset:u32,occupancy:SvoBrickOccupancy)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  // Hoisted for the same reason the DDAs hoist it: every cell scanned below is a
  // cell of this one leaf, and under the banded payload everything but the
  // occupancy bit is a property of the leaf rather than of the cell.
  ${cellIdentitySourceWGSL("voxelOffset")}
  for(var macroBit=0u;macroBit<8u;macroBit+=1u){
    if(occupancy.ready!=0u&&(occupancy.macroMask&(1u<<macroBit))==0u){continue;}
    let origin=vec3u(macroBit&1u,(macroBit>>1u)&1u,(macroBit>>2u)&1u)*4u;
    for(var index=0u;index<64u;index+=1u){
      let local=origin+vec3u(index&3u,(index>>2u)&3u,(index>>4u)&3u);
      if(any(local>=vec3u(brickSize))){continue;}
      let address=svoBrickVoxelIndex(voxelOffset,local,brickSize);
      if(address<dryVoxelCapacity()){
        ${cellSolidGateWGSL("address", "if(dryLodCellSolid(identity)){return identity;}")}
      }
    }
  }
  return 0u;
}
/** First solid cell of one aggregate, scanned along the ray. Aggregate-local DDA. */
fn dryLodAggregateIdentity(ro:vec3f,rd:vec3f,brickMinimum:vec3f,extent:vec3f,voxelOffset:u32,
  cellMinimum:vec3u,stride:u32,tEnter:f32,tExit:f32)->u32{
  let brickSize=max(dry.mapping.brickSize,1u);
  ${cellIdentitySourceWGSL("voxelOffset")}
  let limit=vec3i(cellMinimum+vec3u(stride));
  var entry=max(tEnter,0.0);let point=ro+rd*(entry+1e-5);
  var cell=clamp(vec3i(floor((point-brickMinimum)/extent)),vec3i(cellMinimum),limit-vec3i(1));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=brickMinimum+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  // A DDA through an s^3 subgrid crosses at most 3s-2 cells; 24 covers s=8.
  for(var iteration=0u;iteration<24u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=limit)||entry>tExit){break;}
    let index=svoBrickVoxelIndex(voxelOffset,vec3u(cell),brickSize);
    if(index<dryVoxelCapacity()){
      ${cellSolidGateWGSL("index", "if(dryLodCellSolid(identity)){return identity;}")}
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return 0u;
}
// The intermediate rung: step the brick at stride cells and shade the
// aggregate that stops the ray.
//
// The bounds are the brick's own box and never the instance's occupied sub-AABB.
// The distinction is load-bearing and was the second half of the LOD tier's
// damage: cell coordinates derived from the sub-AABB were fed straight to
// svoBrickVoxelIndex as full-brick indices, so a brick whose occupancy filled
// cells [4..7]^3 rendered a squeezed copy of the whole brick's contents into
// that corner. Callers clip the *interval* with the proxy and pass the box.
//
// What the aggregate cannot do yet is answer without descending: occupancy is
// aggregated (macroMask, free in the node's flags word) but identity is not,
// so an occupied aggregate still scans its own cells for the first solid one.
// The saving is therefore empty space — the mask rejects a 4^3 region in one
// bit test where the fine walk paid a payload fetch per cell — and the shading
// is exact-material-at-aggregate-resolution rather than an invented average.
// One representative identity per macrocell (8 u32 per brick, 134 KB at the
// hero's 4 187) is what would make an occupied aggregate free as well.
fn dryPrimaryLeafAggregateHit(ro:vec3f,rd:vec3f,bounds:mat2x3f,tEnter:f32,tExit:f32,
  voxelOffset:u32,occupancy:SvoBrickOccupancy,stride:u32)->DryHit{
  let brickSize=max(dry.mapping.brickSize,1u);
  let extent=(bounds[1]-bounds[0])/f32(brickSize);
  let aggregates=max(brickSize/max(stride,1u),1u);let aggregateExtent=extent*f32(stride);
  var entry=max(tEnter,0.0);let point=ro+rd*(entry+1e-5);
  var block=clamp(vec3i(floor((point-bounds[0])/aggregateExtent)),vec3i(0),vec3i(i32(aggregates)-1));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=bounds[0]+(vec3f(block)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*aggregateExtent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(DRY_MISS),abs(aggregateExtent/rd),abs(rd)>vec3f(1e-9));
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(block<vec3i(0))||any(block>=vec3i(i32(aggregates)))||entry>tExit){break;}
    let blockExit=min(min(nextT.x,nextT.y),min(nextT.z,tExit));
    let cellMinimum=vec3u(block)*stride;
    // Free rejection: the producer publishes a 4^3 macrocell mask in the
    // terminal node's spare flags word, and an aggregate of four cells or fewer
    // lies wholly inside one macrocell because the grids are aligned and the
    // strides divide. A whole-brick aggregate spans all eight macrocells, so it
    // must consult the occupied bit instead — testing macrocell zero there
    // would reject any brick whose contents sit in another octant.
    let aggregateOccupied=select(svoBrickMacroOccupied(occupancy,cellMinimum),
      occupancy.ready==0u||occupancy.occupied!=0u,stride>4u);
    if(aggregateOccupied){
      let identity=dryLodAggregateIdentity(ro,rd,bounds[0],extent,voxelOffset,cellMinimum,stride,entry,blockExit);
      if(identity!=0u){
        let blockBounds=mat2x3f(bounds[0]+vec3f(cellMinimum)*extent,
          bounds[0]+vec3f(cellMinimum+vec3u(stride))*extent);
        return dryPrimaryVoxelProxyHit(ro,rd,blockBounds,entry,identity);
      }
    }
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){block.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){block.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){block.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
// LOD-tier entry: brick box from the node, interval already clipped by the proxy.
//
// A sub-threshold brick is drawn as one voxel, and the emphasis is on *drawn*:
// this arm may not answer "nothing". It keeps exactly one candidate per pixel —
// the coverage pass depth-tests the proxy boxes and the winner is the only
// instance the resolve is ever handed — and every brick behind that winner was
// routed away from the exact arena for being sub-threshold as well. So a miss
// here is not "try the next brick", it is a hole straight through to terrain or
// sky, which is what the LOD slider was producing at every threshold high
// enough to collapse overlapping geometry.
//
// Over-covering the proxy box instead is the error the budget already names:
// the box only exists because it measured at or under the threshold, so drawing
// it solid can grow a silhouette by at most that many pixels, and at the
// contract's default of three it is invisible. Descending still runs first and
// still picks the material, because a proxy box is tens of pixels across at a
// high threshold and the along-ray cell beats one representative for all of it.
fn dryPrimaryLeafProxyHit(ro:vec3f,rd:vec3f,nodeIndex:u32,proxyBounds:mat2x3f,tEnter:f32,tExit:f32,voxelOffset:u32)->DryHit{
  let node=svoNodeLoad(nodeIndex);
  if(node.links.z!=SVO_INVALID){let leaf=svoLeafLoad(node.links.z);
    if(leaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){
      return dryPlanarTerminalHit(ro,rd,nodeIndex,tEnter,tExit);
    }
  }
  let bounds=svoNodeBounds(node,dry.mapping);
  let occupancy=svoBrickOccupancyDecode(node.links.w);
  let stride=dryLodCellStride(bounds,node.address.z);
  let descended=dryPrimaryLeafAggregateHit(ro,rd,bounds,tEnter,tExit,voxelOffset,occupancy,stride);
  if(descended.t<DRY_MISS){return descended;}
  let identity=dryLodBrickRepresentativeIdentity(voxelOffset,occupancy);
  if(identity==0u){return missHit();}
  return dryPrimaryVoxelProxyHit(ro,rd,proxyBounds,max(tEnter,0.0),identity);
}
fn dryPrimaryPrimitiveProxyHit(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,span:vec2f)->DryHit{
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let extent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
  let bounds=mat2x3f(centre-extent,centre+extent);let t=mix(span.x,span.y,0.5);
  return DryHit(t,dryPrimaryProxyNormal(bounds,ro+rd*t),svoPrimitiveMaterialId(record),
    svoPrimitiveOwnerId(record),SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,
    DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
}
` : "";
  const tieredComputeResolveDeclarationsWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
// W2 scene-primitive compute tier. The four color planes are storage-written
// only for winning pixels; a tiny depth-only bridge commits reversed-Z after
// the compute pass, avoiding another full 48-byte MRT load/store cycle.
@group(${splitGroup + 1}) @binding(0) var dryTierCurrentDepth:texture_depth_2d;
@group(${splitGroup + 1}) @binding(1) var dryTierPackedSurfaceWrite:texture_storage_2d<rgba32uint,write>;
@group(${splitGroup + 1}) @binding(2) var dryTierIdentityMediaWrite:texture_storage_2d<rgba16uint,write>;
@group(${splitGroup + 1}) @binding(3) var dryTierGeometryWrite:texture_storage_2d<rgba32float,write>;
@group(${splitGroup + 1}) @binding(4) var dryTierOpaqueIdentityWrite:texture_storage_2d<rg32uint,write>;
@group(${splitGroup + 1}) @binding(5) var dryTierDepthWrite:texture_storage_2d<r32float,write>;
struct DryTieredResolveQueue{
  dispatchX:atomic<u32>,dispatchY:atomic<u32>,dispatchZ:atomic<u32>,pixelCount:atomic<u32>,pixels:array<u32>,
}
@group(${splitGroup + 1}) @binding(8) var<storage,read_write> dryTieredResolveQueue:DryTieredResolveQueue;
` : "";
  const sceneCoverageReturnTypeWGSL = screenSpaceTerminationPixels > 0
    ? "->DryBrickCoverageOut" : "->@location(0) u32";
  const sceneCoverageMissWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 0u;";
  const sceneCoverageOverflowWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 1u;";
  const screenSpaceSceneCoverageWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let worldExtent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
  if(dryPrimaryBoundsSubPixel(mat2x3f(centre-worldExtent,centre+worldExtent))){
    return DryBrickCoverageOut(input.primitiveIndex+1u,dryHardwareDepth(span.x,rd,camera[1]));
  }
  let resolvedDepth=textureLoad(dryRasterPrimaryGeometryRead,vec2i(coordinate),0).w;
  let resolvedMetadata=textureLoad(dryRasterPrimaryIdentityRead,vec2i(coordinate),0).y;
  if(((resolvedMetadata>>20u)&15u)==DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY
    &&(resolvedMetadata&0xffffu)==svoPrimitiveOwnerId(record)){${sceneCoverageMissWGSL}}
  if(!(span.x<resolvedDepth)){${sceneCoverageMissWGSL}}
` : "";
  // Retired under voxels-only, and the reason is worth keeping.
  //
  // This per-cell sub-pixel test sat *inside* the owner-range branch, so once
  // the walk accepted solidity as `material != 0` every ownerless cell reached
  // `primaryVoxelSurfaceWGSL` without ever being asked whether it was
  // sub-pixel. Hoisting it out would have been the obvious repair, and it buys
  // nothing: what it saved was the analytic `primitiveHit` march, and under
  // voxels-only there is no march — both arms return at the first solid cell,
  // and the proxy arm returns the *same* cell with a coarser depth. A test that
  // costs a footprint evaluation per cell to choose between two identical
  // answers is a cost, not a level of detail.
  //
  // Real intermediate detail is the aggregate stride (`primaryLodStrideWGSL`),
  // decided once per brick instead of once per cell. The control arm that still
  // marches records keeps the original test, where it still earns its place.
  const screenSpacePrimitiveResolveWGSL = screenSpaceTerminationPixels > 0
    ? "let proxyLocalExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));let proxyExtent=dryScenePrimitiveWorldExtent(proxyLocalExtent,record.orientation);let proxyCentre=svoPrimitiveCenter_m(record);if(dryPrimaryBoundsSubPixel(mat2x3f(proxyCentre-proxyExtent,proxyCentre+proxyExtent))){let proxySpan=vec2f(span.x,min(span.y,limit));return dryPrimaryPrimitiveProxyHit(record,ro,rd,proxySpan);}"
    : "";
  const brickCoverageOutputWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
struct DryBrickCoverageOut{@location(0) key:u32,@builtin(frag_depth) hardwareDepth:f32}
` : "";
  const brickCoverageReturnTypeWGSL = screenSpaceTerminationPixels > 0
    ? "->DryBrickCoverageOut" : "->@location(0) u32";
  const brickCoverageMissWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 0u;";
  const brickCoverageOverflowWGSL = screenSpaceTerminationPixels > 0
    ? "return DryBrickCoverageOut(0u,0.0);" : "return 1u;";
  const screenSpaceBrickCoverageWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let proxyBounds=mat2x3f(input.proxyMinimum,input.proxyMaximum);
  if(dryPrimaryBrickProxySubPixel(proxyBounds,dryLeafLevel(input.nodeIndex))){
    let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
    let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,proxyBounds);
    if(interval.x!=0.0){let entry=max(interval.y,0.0);return DryBrickCoverageOut(input.instanceIndex+1u,dryHardwareDepth(entry,rd,camera[1]));}
  }
` : "";
  const screenSpaceBrickResolveWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  let lodKey=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(lodKey>0u){let instanceIndex=lodKey-1u;if(instanceIndex<arrayLength(&svoBrickInstances)){
    let record=svoBrickInstances[instanceIndex];let proxyBounds=mat2x3f(record.proxyMinimum,record.proxyMaximum);
    let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,proxyBounds);
    if(interval.x!=0.0){let proxy=dryPrimaryLeafProxyHit(ro,rd,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,proxyBounds,max(interval.y,0.0),interval.z,record.voxelOffset);
      if(proxy.t<opaque.t){opaque=proxy;producer=SVO_GBUFFER_PRODUCER_BRICK;}}
  }}
` : "";
  // The thresholded production graph resolves background, LOD and exact
  // surfaces in separate passes, so it has no single full-screen writer for
  // the old candidate-arena depth seed. Hardware depth remains authoritative;
  // omit this optional early-out instead of consulting stale arena words.
  const sceneCoverageSeedCullWGSL = screenSpaceTerminationPixels > 0 ? "" : /* wgsl */ `
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)
    &&!(span.x<bitcast<f32>(svoBrickCoverageCandidates[seedIndex]))){return 0u;}
`;
  const sceneCoverageSeedLimitWGSL = screenSpaceTerminationPixels > 0 ? "" : /* wgsl */ `
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)){limit=bitcast<f32>(svoBrickCoverageCandidates[seedIndex]);}
`;
  const traversalCursorWGSL = canonicalTraversal ? /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);}
` : secondaryTraversalMode === "compact" ? /* wgsl */ `
struct DryTraversalCursor{compact:SvoCompactTraversalContinuation}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){svoCompactContinuationBegin(ray,mapping,&(*cursor).compact);}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoCompactContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).compact);}
` : secondaryTraversalMode === "wide" ? /* wgsl */ `
struct DryTraversalCursor{wide:SvoWideTraversalCursor}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return dryPublicationWord(2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){let initialized=svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration());if(!initialized){(*cursor).wide.state=SVO_WIDE_CURSOR_INVALID;}}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{return svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());}
` : /* wgsl */ `
struct DryTraversalCursor{canonical:SvoTraversalContinuation,wide:SvoWideTraversalCursor,useWide:u32}
fn dryWidePublication()->SvoWidePublication{return SvoWidePublication(dry.wideFanout.x,dry.wideFanout.y,dry.wideFanout.z,dry.wideFanout.w);}
fn dryCanonicalPublicationGeneration()->u32{return dryPublicationWord(2u);}
fn dryTraversalCursorBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>){
  (*cursor).useWide=select(0u,1u,svoWideCursorInitialize(&(*cursor).wide,ray,mapping,dryWidePublication(),dryCanonicalPublicationGeneration()));
  if((*cursor).useWide==0u){svoTraversalContinuationBegin(ray,mapping,&(*cursor).canonical);}
}
fn dryTraversalCursorNext(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,DryTraversalCursor>)->SvoTraversalHit{
  if((*cursor).useWide!=0u){let wideHit=svoWideCursorNext(&(*cursor).wide,ray,mapping,dryDiagnosticMaximumDepth(),dryWidePublication(),dryCanonicalPublicationGeneration());if(wideHit.status==SVO_STATUS_HIT||wideHit.status==SVO_STATUS_MISS||wideHit.status==SVO_STATUS_WORK_EXHAUSTED){return wideHit;}(*cursor).useWide=0u;if(wideHit.visits>=mapping.maxVisits){return svoMiss(SVO_STATUS_WORK_EXHAUSTED,wideHit.visits);}var fallbackMapping=mapping;fallbackMapping.maxVisits-=wideHit.visits;svoTraversalContinuationBegin(ray,fallbackMapping,&(*cursor).canonical);var fallback=svoTraversalContinuationNext(ray,fallbackMapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);fallback.visits+=wideHit.visits;return fallback;}
  return svoTraversalContinuationNext(ray,mapping,dryDiagnosticMaximumDepth(),&(*cursor).canonical);
}
`;
  const brickOccupancyHelpersWGSL = /* wgsl */ `${svoBrickOccupancyWGSL}
${brickContour ? svoBrickContourWGSL : ""}
${brickOccupancyMode === "off" ? "" : /* wgsl */ `
const DRY_BRICK_OCCUPANCY_MACRO:u32=${brickOccupancyMode === "macro" ? 1 : 0}u;
fn dryBrickMacroSkip(summary:SvoBrickOccupancy,local:vec3u,bounds:mat2x3f,extent:vec3f,ro:vec3f,rd:vec3f,entry:f32)->vec2f{
  if(DRY_BRICK_OCCUPANCY_MACRO==0u||svoBrickMacroOccupied(summary,local)){return vec2f(0.0,entry);}
  let macroCoord=local>>vec3u(2u);let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let boundaryCell=vec3i(macroCoord*4u)+select(vec3i(0),vec3i(4),step>vec3i(0));
  let boundary=bounds[0]+vec3f(boundaryCell)*extent;
  let next=select(vec3f(3.402823e38),(boundary-ro)/rd,abs(rd)>vec3f(1e-9));
  return vec2f(1.0,max(entry,min(next.x,min(next.y,next.z))));
}
`}`;
  // One node record per leaf visit, not three.
  //
  // `dryLeafBounds`, `dryLeafFlags` and `dryLeafLevel` each expand to a whole
  // `svoNodeLoad`, and under the arena form that is eight scalar `svoStructure`
  // reads apiece (webgpu-svo-traversal.ts). The leaf walk called all three on
  // the same `hit.nodeIndex` — up to 24 scalar storage loads of one 32-byte
  // record per visit, against a primary budget of 48 leaf visits. Load once and
  // read the fields off the record, which is what `dryPrimaryLeafProxyHit`
  // already does.
  //
  // The compact arm keeps `dryLeafBounds`: its bounds come from a different
  // buffer (`svoCompactNodes`), so there the canonical record is loaded only
  // when the flags or level word is actually wanted.
  const leafNodeRecordNeeded = brickOccupancyMode !== "off" || screenSpaceTerminationPixels > 0
    || brickContour;
  // The visit's own record where the cursor published one; a fresh fetch of the
  // same 32 bytes otherwise. `svoNodeBounds` reads the same fields either way,
  // so the bounds — and the DDA seeded from them — are bit-identical.
  const leafNodeSetupWGSL = compactTraversal
    ? `${leafNodeRecordNeeded ? "let leafNode=svoNodeLoad(hit.nodeIndex);" : ""}let bounds=dryLeafBounds(hit.nodeIndex);`
    : primaryVisitRecords
      ? "let leafNode=visitNode;let bounds=svoNodeBounds(leafNode,dry.mapping);"
      : "let leafNode=svoNodeLoad(hit.nodeIndex);let bounds=svoNodeBounds(leafNode,dry.mapping);";
  /** The trailing record parameter the leaf payload walk takes on that arm. */
  const visitNodeParameterWGSL = primaryVisitRecords ? ",visitNode:SvoNode" : "";
  // Where the DDA is allowed to step, published so the loop escape test reads
  // it rather than the whole brick.
  //
  // `bounds` mode already clips the ray interval to the occupied sub-AABB but
  // still walked cells across the dead margin, because the clamp and the escape
  // test were both written against `[0, brickSize-1]`. The summary carries the
  // exact sub-box, and it is in *full-brick* local coordinates — which is the
  // only form `svoBrickVoxelIndex` accepts — so clamping to it is sound without
  // rebasing `cell`. Every cell it removes is empty by the summary's own
  // construction, so no nearer hit can be lost.
  //
  // `macro` keeps the full-brick span deliberately: its per-step skip re-clamps
  // `cell` to `[0, brickSize-1]` after every jump, and a sub-box escape test
  // would turn float slop at the sub-box face into an early break. That arm is
  // a measured +33% regression at depth 3 and is left exactly as it was.
  const brickCellSpanClamped = brickOccupancyMode === "bounds";
  const fullBrickCellSpanWGSL = "let cellMinimum=vec3i(0);let cellMaximum=vec3i(i32(dry.mapping.brickSize-1u));";
  // The contour clamp, spliced into exactly the place `bounds` already clamps
  // and reading the record `bounds` already loaded. `svoBrickContourClamp` wants
  // the ray in brick cell units, which is `(point - bounds[0]) / extent` — the
  // same change of variables the DDA seed on the next line performs.
  //
  // Both arms need a mutable `brickExit`, which only the occupancy arms declare;
  // `brickIntervalDeclared` is what carries that through to the escape test.
  const brickIntervalDeclared = brickOccupancyMode !== "off" || brickContour;
  // Raising `entry` is the one part of an entry-interval clamp that is not
  // image-exact, and the reason is structural rather than an epsilon.
  //
  // The DDA reports a hit at `entry`, and `entry` is not recomputed per cell —
  // it is the running `min(nextT)` of an *incrementally accumulated* boundary
  // vector (`nextT.x += deltaT.x`). Seeding the walk at a later cell therefore
  // performs a different number of accumulations to reach the same face, and
  // float addition is not associative: the same geometric crossing comes back
  // one ULP apart. Measured on the hero garden at depth 3, that is ~1 % of
  // pixels differing by one f16 ULP — no pixel gains or loses a surface, and no
  // brick is wrongly rejected. It is nonetheless not byte-identical, and it is
  // exactly why `bounds` "moves the image" while being provably conservative.
  //
  // Rejecting the brick and clamping `brickExit` have no such exposure: a
  // rejected brick had no solid cell on the chord and the walk returned a miss
  // anyway, and every solid cell's own exit is inside the slab by construction,
  // so the shortened `brickExit` never reaches a `cellExit` that mattered. That
  // is the default here, and the entry raise WAS the opt-in.
  //
  // It is now on by default, because the measurement inverted the trade: the
  // exit clamp and rejection together are worth ~0% and the entry raise is worth
  // the entire -19.2%. Defaulting the safe half alone ships the exposure of a
  // feature with none of its benefit.
  //
  // The exposure is also smaller than the framing above implies, and is not a
  // lost-surface risk. The difference it produces is the **shader compiler**, not
  // the clamp: emitting the whole decode and interval test but consuming it in a
  // branch that can never be taken is byte-identical, while clamping only the
  // *dead* `cellExit` — never read under `voxelsOnlyPrimary` — reproduces the
  // difference exactly, hash for hash and ULP for ULP. So does a control that
  // reads a variable the contour never writes. Dawn/Metal reassociates float
  // arithmetic in the DDA loop once the loop body's dataflow changes shape; a
  // 1-ULP shift in `entry` flips `dryVoxelFaceNormal`'s argmin on a cell-edge
  // pixel, which is why the mask is scattered isolated pixels on lit ground and
  // canopy with no coherent holes, silhouette band, or horizon structure.
  //
  // The consequence outlives this flag: **a byte-identical settled-frame hash is
  // not a valid acceptance oracle for any edit to the primary DDA loop body**,
  // however sound the edit. Use full-population escape count plus a bounded
  // per-pixel radiance delta instead. That also dissolves the apparent
  // contradiction in the depth-3 handoff doc — the byte-identical result was for
  // the fused leaf-node loads, which do not touch the loop body, and the
  // "moves the image" result was the span clamp, which does. Both were true.
  const brickContourEntryClamp = brickContour && experiments.brickContourEntryClamp !== false;
  const brickContourInertProbe = experiments.brickContourInertProbe === true;
  // `brickExit` reaches the walk through exactly two expressions — the loop
  // escape and `cellExit` — and they are not the same claim. The escape decides
  // which cells are *tested*; `cellExit` only bounds the owner-run interval,
  // which voxels-only never solves. Splitting them is what turns "the clamp
  // changed the image" into a statement about one of the two.
  const brickContourExitScope = experiments.brickContourExitScope ?? "both";
  const brickContourPrimary = brickContour && experiments.brickContourVisibilityOnly !== true;
  const brickContourVisibility = brickContour && experiments.brickContourPrimaryOnly !== true;
  const contourClampWGSL = (origin: string, direction: string, miss: string, enabled = true) => brickContour && enabled
    ? /* wgsl */ `
  {let contour=svoBrickContourDecode(leafNode.address.w);
  if(contour.valid!=0u){
    let contourSpan=svoBrickContourClamp(contour,(${origin}-bounds[0])/extent,${direction}/extent,entry,brickExit);
    if(contourSpan.x==${brickContourInertProbe ? "2.0" : "0.0"}){return ${miss};}
    ${brickContourInertProbe ? "" : `${brickContourEntryClamp ? "entry=contourSpan.y;" : ""}${brickContourExitScope === "cell-exit" ? "brickCellExit" : brickContourExitScope === "cell-exit-inert" ? "brickExitUnused" : "brickExit"}=contourSpan.z;`}}}`
    : "";
  const primaryBrickSetupWGSL = brickOccupancyMode === "off"
    ? /* wgsl */ `${leafNodeSetupWGSL} let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  ${fullBrickCellSpanWGSL}
  var entry=max(hit.tEnter,0.0);${brickContour ? "var brickExit=hit.tExit;" : ""}${brickContourExitScope === "cell-exit" || brickContourExitScope === "cell-exit-inert" ? "var brickCellExit=hit.tExit;var brickExitUnused=hit.tExit;" : ""}${contourClampWGSL("ro", "rd", "missHit()", brickContourPrimary)}
  let point=ro+rd*(entry+1e-5); var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum)));`
    : /* wgsl */ `${leafNodeSetupWGSL}let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  let brickSummary=svoBrickOccupancyDecode(leafNode.links.w);var brickExit=hit.tExit;${brickContourExitScope === "cell-exit" || brickContourExitScope === "cell-exit-inert" ? "var brickCellExit=hit.tExit;var brickExitUnused=hit.tExit;" : ""}var entry=max(hit.tEnter,0.0);
  ${brickCellSpanClamped ? "var cellMinimum=vec3i(0);var cellMaximum=vec3i(i32(dry.mapping.brickSize-1u));" : fullBrickCellSpanWGSL}
  if(brickSummary.ready!=0u){if(brickSummary.occupied==0u){return missHit();}let occupiedInterval=svoRayAabbWithInverse(SvoRay(ro,entry,rd,brickExit),1.0/rd,svoBrickOccupiedBounds(brickSummary,bounds[0],extent));if(occupiedInterval.x==0.0){return missHit();}entry=max(entry,occupiedInterval.y);brickExit=min(brickExit,occupiedInterval.z);${brickCellSpanClamped ? "cellMinimum=vec3i(brickSummary.minInclusive);cellMaximum=vec3i(brickSummary.maxInclusive);" : ""}}${contourClampWGSL("ro", "rd", "missHit()", brickContourPrimary)}
  let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum)));`;
  const primaryBrickExitWGSL = brickIntervalDeclared ? "brickExit" : "hit.tExit";
  const primaryCellExitWGSL = brickContourExitScope === "cell-exit" || brickContourExitScope === "cell-exit-inert" ? "brickCellExit" : primaryBrickExitWGSL;
  const primaryMacroSkipWGSL = brickOccupancyMode === "macro" ? /* wgsl */ `let macroSkip=dryBrickMacroSkip(brickSummary,vec3u(cell),bounds,extent,ro,rd,entry);if(macroSkip.x!=0.0){if(macroSkip.y>=brickExit||macroSkip.y>=DRY_MISS){break;}entry=macroSkip.y;let skipPoint=ro+rd*(entry+max(1e-5,length(extent)*1e-4));cell=vec3i(clamp(floor((skipPoint-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));let skipBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;nextT=select(vec3f(DRY_MISS),(skipBoundary-ro)/rd,abs(rd)>vec3f(1e-9));continue;}
    ` : "";
  // The shadow twin of the two notes above: one node record, and the same
  // `bounds`-mode cell span. The visibility walk has no LOD-stride fragment, so
  // its `off` arm never needs the record at all.
  const shadowLeafNodeSetupWGSL = compactTraversal
    ? `${brickIntervalDeclared ? "let leafNode=svoNodeLoad(hit.nodeIndex);" : ""}let bounds=dryLeafBounds(hit.nodeIndex);`
    : "let leafNode=svoNodeLoad(hit.nodeIndex);let bounds=svoNodeBounds(leafNode,dry.mapping);";
  const shadowContourMissWGSL = "dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS)";
  const shadowBrickSetupWGSL = brickOccupancyMode === "off"
    ? /* wgsl */ `${shadowLeafNodeSetupWGSL}let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  ${fullBrickCellSpanWGSL}
  var entry=max(max(hit.tEnter,tMin_m),0.0);${brickContour ? "var brickExit=hit.tExit;" : ""}${contourClampWGSL("ray.origin_m", "ray.direction", shadowContourMissWGSL, brickContourVisibility)}
  let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum)));`
    : /* wgsl */ `${shadowLeafNodeSetupWGSL}let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);
  let brickSummary=svoBrickOccupancyDecode(leafNode.links.w);var brickExit=min(hit.tExit,ray.tMax_m);var entry=max(max(hit.tEnter,tMin_m),0.0);
  ${brickCellSpanClamped ? "var cellMinimum=vec3i(0);var cellMaximum=vec3i(i32(dry.mapping.brickSize-1u));" : fullBrickCellSpanWGSL}
  if(brickSummary.ready!=0u){if(brickSummary.occupied==0u){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}let occupiedInterval=svoRayAabbWithInverse(SvoRay(ray.origin_m,entry,ray.direction,brickExit),1.0/ray.direction,svoBrickOccupiedBounds(brickSummary,bounds[0],extent));if(occupiedInterval.x==0.0){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}entry=max(entry,occupiedInterval.y);brickExit=min(brickExit,occupiedInterval.z);${brickCellSpanClamped ? "cellMinimum=vec3i(brickSummary.minInclusive);cellMaximum=vec3i(brickSummary.maxInclusive);" : ""}}${contourClampWGSL("ray.origin_m", "ray.direction", shadowContourMissWGSL, brickContourVisibility)}
  let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum)));`;
  const shadowBrickExitWGSL = brickIntervalDeclared ? "brickExit" : "hit.tExit";
  const shadowMacroSkipWGSL = brickOccupancyMode === "macro" ? /* wgsl */ `let macroSkip=dryBrickMacroSkip(brickSummary,vec3u(cell),bounds,extent,ray.origin_m,ray.direction,entry);if(macroSkip.x!=0.0){if(macroSkip.y>=brickExit||macroSkip.y>=DRY_MISS){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}entry=macroSkip.y;let skipPoint=ray.origin_m+ray.direction*(entry+max(1e-5,length(extent)*1e-4));cell=vec3i(clamp(floor((skipPoint-bounds[0])/extent),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));let skipBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;nextT=select(vec3f(DRY_MISS),(skipBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));continue;}
    ` : "";
  // Where the exact tier becomes adaptive.
  //
  // Routing the LOD by *pass* — sub-threshold pixels to the proxy fragment,
  // the rest to the exact one — only ever offered two granularities, and once
  // the whole-brick predicate was corrected to measure the whole brick it
  // stopped firing on anything but far background. The intermediate rungs
  // therefore have to live inside the exact walk, decided once per brick
  // fragment from the projected footprint. That is also the cheapest place for
  // it: `bounds` and the occupancy word are already in hand.
  //
  // Stride one leaves the loop below untouched, byte for byte, which is what
  // keeps a zero threshold the reference image rather than a near-miss of it.
  const primaryLodStrideWGSL = screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  // Capped one rung above the whole brick, and that cap is not a taste call: an
  // aggregate the size of the brick still scans every cell the ray crosses to
  // find a material, so it pays the exact walk's cost and then discards the
  // precision it bought. The whole-brick answer is cheap only where it can be
  // given without touching the payload — the coverage pass, which returns the
  // proxy face and a hardware depth and stops. Here it is a pure loss.
  //
  // Level and flags come off the record the brick setup already loaded — this
  // fragment is spliced into the same function body, so leafNode is in scope —
  // rather than from two more whole-record loads of the same 32 bytes.
  let lodStride=min(dryLodCellStride(bounds,leafNode.address.z),max(dry.mapping.brickSize>>1u,1u));
  if(lodStride>1u){
    return dryPrimaryLeafAggregateHit(ro,rd,bounds,entry,${primaryBrickExitWGSL},hit.voxelOffset,
      svoBrickOccupancyDecode(leafNode.links.w),lodStride);
  }` : "";
  const primaryLeafVoxelTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafVoxelPayloadMacroHdda" : "traceLeafVoxelPayload";
  // The resolver reads five words out of two records the cursor has just read.
  // Taking them as parameters is what turns the second fetch of each into
  // nothing; the tests it applies are unchanged, including the two the cursor
  // has already made — they cost a component extract off one control vector,
  // and dropping them would make this function's contract depend on the
  // caller's, which is the kind of coupling a publication bug hides in.
  const primaryLeafResolveSignatureWGSL = primaryVisitRecords
    ? "fn dryPrimaryLeafResolve(nodeIndex:u32,node:SvoNode,leaf:SvoLeaf)->DryLeafResolution{"
    : "fn dryPrimaryLeafResolve(nodeIndex:u32)->DryLeafResolution{\n  let node=svoNodeLoad(nodeIndex);";
  const primaryLeafResolveLeafLoadWGSL = primaryVisitRecords ? "" : "let leaf=svoLeafLoad(leafIndex);";
  /** Taken once, immediately after the cursor call that wrote them. */
  const primaryVisitRecordTakeWGSL = primaryVisitRecords
    ? "let visitNode=svoVisitNode;let visitLeaf=svoVisitLeaf;" : "";
  const primaryLeafResolveCallWGSL = primaryVisitRecords
    ? "dryPrimaryLeafResolve(leaf.nodeIndex,visitNode,visitLeaf)" : "dryPrimaryLeafResolve(leaf.nodeIndex)";
  const shadowLeafTraceCallWGSL = brickOccupancyMode === "macro-hdda" ? "traceLeafPayloadVisibilityMacroHdda" : "traceLeafPayloadVisibility";
  const macroHddaPrimaryWGSL = brickOccupancyMode === "macro-hdda" ? /* wgsl */ `
fn traceLeafPayloadFineInterval(ro:vec3f,rd:vec3f,hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,intervalEnter:f32,intervalExit:f32,cellMinimum:vec3u,cellMaximum:vec3u)->DryHit{
  var entry=max(intervalEnter,0.0);let point=ro+rd*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum-vec3u(1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));let tolerance=length(extent)*1.05;
  ${cellIdentitySourceWGSL("hit.voxelOffset")}
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=vec3i(cellMaximum))||entry>intervalExit){break;}
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<dryVoxelCapacity()){${cellSolidGateWGSL("payloadIndex", "let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);let faceNormal=dryVoxelFaceNormal(cellBounds,ro+rd*entry);let shaded=dryShadingNormal(identity,faceNormal);let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,intervalExit));let surfaceT=drySmoothVoxelSurfaceT(identity,payloadIndex,cellBounds,ro,rd,entry,cellExit,shaded.normal);return DryHit(surfaceT,shaded.normal,sceneIdentityMaterial(identity),DRY_OWNER_NONE,shaded.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));")}}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
fn traceLeafVoxelPayloadMacroHdda(ro:vec3f,rd:vec3f,hit:SvoTraversalHit${visitNodeParameterWGSL})->DryHit{
  ${leafNodeSetupWGSL}let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);let summary=svoBrickOccupancyDecode(leafNode.links.w);
  if(summary.ready==0u){return traceLeafPayloadFineInterval(ro,rd,hit,bounds,extent,hit.tEnter,hit.tExit,vec3u(0u),vec3u(dry.mapping.brickSize));}
  if(summary.occupied==0u){return missHit();}
  let interval=svoRayAabbWithInverse(SvoRay(ro,max(hit.tEnter,0.0),rd,hit.tExit),1.0/rd,svoBrickOccupiedBounds(summary,bounds[0],extent));if(interval.x==0.0){return missHit();}
  var macroEntry=max(max(hit.tEnter,interval.y),0.0);let brickExit=min(hit.tExit,interval.z);let macroExtent=extent*4.0;let point=ro+rd*(macroEntry+1e-5);var macroCell=vec3i(clamp(floor((point-bounds[0])/macroExtent),vec3f(0.0),vec3f(1.0)));
  let macroStep=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));let macroBoundary=bounds[0]+(vec3f(macroCell)+select(vec3f(0.0),vec3f(1.0),macroStep>vec3i(0)))*macroExtent;
  var macroNextT=select(vec3f(DRY_MISS),(macroBoundary-ro)/rd,abs(rd)>vec3f(1e-9));let macroDeltaT=select(vec3f(DRY_MISS),abs(macroExtent/rd),abs(rd)>vec3f(1e-9));
  for(var macroIteration=0u;macroIteration<8u;macroIteration+=1u){
    if(any(macroCell<vec3i(0))||any(macroCell>=vec3i(2))||macroEntry>brickExit){break;}let macroExit=min(min(macroNextT.x,macroNextT.y),min(macroNextT.z,brickExit));let macroCoord=vec3u(macroCell);let macroBit=macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u);
    if((summary.macroMask&(1u<<macroBit))!=0u){let cellMinimum=macroCoord*4u;let candidate=traceLeafPayloadFineInterval(ro,rd,hit,bounds,extent,macroEntry,macroExit,cellMinimum,cellMinimum+vec3u(4u));if(candidate.t<DRY_MISS){return candidate;}}
    let advance=min(macroNextT.x,min(macroNextT.y,macroNextT.z));if(macroNextT.x<=advance+1e-6){macroCell.x+=macroStep.x;macroNextT.x+=macroDeltaT.x;}if(macroNextT.y<=advance+1e-6){macroCell.y+=macroStep.y;macroNextT.y+=macroDeltaT.y;}if(macroNextT.z<=advance+1e-6){macroCell.z+=macroStep.z;macroNextT.z+=macroDeltaT.z;}macroEntry=advance;
  }
  return missHit();
}
` : "";
  const macroHddaShadowWGSL = brickOccupancyMode === "macro-hdda" ? /* wgsl */ `
fn traceLeafPayloadVisibilityFineInterval(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,bounds:mat2x3f,extent:vec3f,intervalEnter:f32,intervalExit:f32,cellMinimum:vec3u,cellMaximum:vec3u,workLimit:u32)->SvoVisibilityStep{
  var entry=max(max(intervalEnter,tMin_m),0.0);let point=ray.origin_m+ray.direction*(entry+1e-5);var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(cellMinimum),vec3f(cellMaximum-vec3u(1u))));
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));let tolerance=length(extent)*1.05;var workItems=0u;
  ${cellIdentitySourceWGSL("hit.voxelOffset")}
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(cellMinimum))||any(cell>=vec3i(cellMaximum))||entry>intervalExit||entry>ray.tMax_m){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}workItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);if(payloadIndex>=dryVoxelCapacity()){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
    ${cellSolidGateWGSL("payloadIndex", "let materialId=sceneIdentityMaterial(identity);if(materialId>=dry.materialPublication.x){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}let material=dryMaterial(materialId);if(!dryMaterialPublished(material,materialId)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}if(dryMaterialThinDielectric(material,materialId)){let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);let normal=dryVoxelFaceNormal(cellBounds,ray.origin_m+ray.direction*entry);let cellExit=min(nextT.x,min(nextT.y,nextT.z));return dryVisibilityTransmissionStep(0u,0u,workItems,min(max(cellExit,entry),ray.tMax_m),dryThinDielectricTransmittance(material,normal,ray.direction));}return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,entry);")}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);
}
fn traceLeafPayloadVisibilityMacroHdda(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,workLimit:u32)->SvoVisibilityStep{
  let terminalNode=svoNodeLoad(hit.nodeIndex);
  if(terminalNode.links.z!=SVO_INVALID){let terminalLeaf=svoLeafLoad(terminalNode.links.z);
    if(terminalLeaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){
      if(workLimit==0u){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,0u,DRY_MISS);}
      return dryPlanarTerminalVisibility(ray,hit);
    }
  }
  ${shadowLeafNodeSetupWGSL}let extent=(bounds[1]-bounds[0])/f32(dry.mapping.brickSize);let summary=svoBrickOccupancyDecode(leafNode.links.w);
  if(summary.ready==0u){return traceLeafPayloadVisibilityFineInterval(ray,tMin_m,hit,bounds,extent,hit.tEnter,min(hit.tExit,ray.tMax_m),vec3u(0u),vec3u(dry.mapping.brickSize),workLimit);}if(summary.occupied==0u){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}
  let interval=svoRayAabbWithInverse(SvoRay(ray.origin_m,max(max(hit.tEnter,tMin_m),0.0),ray.direction,min(hit.tExit,ray.tMax_m)),1.0/ray.direction,svoBrickOccupiedBounds(summary,bounds[0],extent));if(interval.x==0.0){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,0u,DRY_MISS);}
  var macroEntry=max(max(hit.tEnter,interval.y),tMin_m);let brickExit=min(min(hit.tExit,interval.z),ray.tMax_m);let macroExtent=extent*4.0;let point=ray.origin_m+ray.direction*(macroEntry+1e-5);var macroCell=vec3i(clamp(floor((point-bounds[0])/macroExtent),vec3f(0.0),vec3f(1.0)));
  let macroStep=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let macroBoundary=bounds[0]+(vec3f(macroCell)+select(vec3f(0.0),vec3f(1.0),macroStep>vec3i(0)))*macroExtent;
  var macroNextT=select(vec3f(DRY_MISS),(macroBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let macroDeltaT=select(vec3f(DRY_MISS),abs(macroExtent/ray.direction),abs(ray.direction)>vec3f(1e-9));var workItems=0u;
  for(var macroIteration=0u;macroIteration<8u;macroIteration+=1u){
    if(any(macroCell<vec3i(0))||any(macroCell>=vec3i(2))||macroEntry>brickExit){break;}let macroExit=min(min(macroNextT.x,macroNextT.y),min(macroNextT.z,brickExit));let macroCoord=vec3u(macroCell);let macroBit=macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u);
    if((summary.macroMask&(1u<<macroBit))!=0u){if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}let cellMinimum=macroCoord*4u;var fine=traceLeafPayloadVisibilityFineInterval(ray,tMin_m,hit,bounds,extent,macroEntry,macroExit,cellMinimum,cellMinimum+vec3u(4u),workLimit-workItems);workItems+=fine.workItems;if(fine.status!=SVO_VIS_STEP_MISS){fine.workItems=workItems;return fine;}}
    let advance=min(macroNextT.x,min(macroNextT.y,macroNextT.z));if(macroNextT.x<=advance+1e-6){macroCell.x+=macroStep.x;macroNextT.x+=macroDeltaT.x;}if(macroNextT.y<=advance+1e-6){macroCell.y+=macroStep.y;macroNextT.y+=macroDeltaT.y;}if(macroNextT.z<=advance+1e-6){macroCell.z+=macroStep.z;macroNextT.z+=macroDeltaT.z;}macroEntry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);
}
` : "";
  const prepassDeclarationsWGSL = reduced ? /* wgsl */ `// Reduced-rate cone-lighting prepass consumption. One integer plane packs
// 8-bit AO plus eight 7-bit light visibilities. Geometry packs distance, an
// octahedral normal, and 11-bit metadata; identity packs uint16 material+owner.
@group(1) @binding(0) var dryPrepassVisibilityKeyTexture:texture_2d<u32>;
@group(1) @binding(1) var dryPrepassGeometryTexture:texture_2d<f32>;
@group(1) @binding(2) var dryPrepassIdentityTexture:texture_2d<u32>;
@group(1) @binding(3) var dryPrepassRadianceTexture:texture_2d<f32>;
var<private> dryPrepassData0:vec4f;
var<private> dryPrepassData1:vec4f;
var<private> dryPrepassData2:vec4f;
var<private> dryPrepassRadiance:vec4f;
var<private> dryPrepassState:u32;
var<private> dryPrepassRadianceState:u32;
var<private> dryPrepassGi:vec4f;
var<private> dryPrepassGiState:u32;
var<private> dryPrepassExactEdgeState:u32;
var<private> dryCurrentLightSlot:u32;
const DRY_PREPASS_INVALID_PACKED:vec2u=vec2u(0xffffffffu,0xfffffffeu);
// Retained only by the retired, unreachable exact-refinement shader entries
// until their source block is removed; no production pipeline compiles them.
const DRY_SILHOUETTE_STATE_UNTOUCHED:u32=0u;
const DRY_SILHOUETTE_STATE_VALID:u32=1u;
const DRY_SILHOUETTE_STATE_FAILED:u32=2u;
fn dryPrepassQuantize7(value:f32)->u32{return u32(round(clamp(value,0.0,1.0)*127.0));}
fn dryPrepassPack(data0:vec4f,data1:vec4f,data2:vec4f)->vec2u{
  let light3=dryPrepassQuantize7(data1.x);
  let word0=u32(round(clamp(data0.x,0.0,1.0)*255.0))|(dryPrepassQuantize7(data0.y)<<8u)|(dryPrepassQuantize7(data0.z)<<15u)|(dryPrepassQuantize7(data0.w)<<22u)|((light3&7u)<<29u);
  let word1=(light3>>3u)|(dryPrepassQuantize7(data1.y)<<4u)|(dryPrepassQuantize7(data1.z)<<11u)|(dryPrepassQuantize7(data1.w)<<18u)|(dryPrepassQuantize7(data2.x)<<25u);return vec2u(word0,word1);
}
fn dryPrepassUnpack0(packed:vec4u)->vec4f{return vec4f(f32(packed.x&255u)/255.0,f32((packed.x>>8u)&127u)/127.0,f32((packed.x>>15u)&127u)/127.0,f32((packed.x>>22u)&127u)/127.0);}
fn dryPrepassUnpack1(packed:vec4u)->vec4f{let light3=((packed.x>>29u)&7u)|((packed.y&15u)<<3u);return vec4f(f32(light3)/127.0,f32((packed.y>>4u)&127u)/127.0,f32((packed.y>>11u)&127u)/127.0,f32((packed.y>>18u)&127u)/127.0);}
fn dryPrepassUnpack2(packed:vec4u)->vec4f{return vec4f(f32((packed.y>>25u)&127u)/127.0,1.0,1.0,1.0);}
fn dryPrepassEncodeNormal(normalIn:vec3f)->vec2f{let normal=normalize(normalIn);var oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));if(normal.z<0.0){oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0.0));}return oct;}
fn dryPrepassDecodeNormal(octIn:vec2f)->vec3f{var normal=vec3f(octIn,1.0-abs(octIn.x)-abs(octIn.y));if(normal.z<0.0){let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));normal=vec3f(folded,normal.z);}return normalize(normal);}
fn dryPrepassHitMetadata(hit:DryHit)->u32{return (hit.featureId&15u)|((hit.fieldSource&15u)<<4u)|((hit.motionKind&3u)<<8u)|((hit.motionValid&1u)<<10u);}
fn dryPrepassPackIdentity(hit:DryHit)->u32{return (hit.materialId&0xffffu)|((hit.ownerId&0xffffu)<<16u);}
fn dryPrepassChannel(index:u32)->f32{
  ${fastDeferred ? "return dryPrepassData0.y;" : `
  if(index<4u){return dryPrepassData0[index];}
  if(index<8u){return dryPrepassData1[index-4u];}
  return dryPrepassData2[min(index-8u,3u)];`}
}
fn dryPrepassReceiverCompatible(identity:u32,metadata:u32,hit:DryHit)->bool{
  let materialMatches=(identity&0xffffu)==(hit.materialId&0xffffu);
  let ownerMatches=(identity>>16u)==(hit.ownerId&0xffffu);
  // Static authored surfaces with the same complete shading classification may
  // share a nearby receiver across object seams. Motion keeps exact ownership:
  // its current-frame rigid blocker correction and GI neighbourhood are owned.
  // The opaque no-GI relight specialization shares visibility, never material
  // colour or radiance; it still evaluates the receiving material at full rate.
  return ${fastDeferred ? "(materialMatches||dry.tuningCounts2.w==4u)" : "materialMatches"}&&metadata==dryPrepassHitMetadata(hit)&&(hit.motionKind==DRY_GBUFFER_MOTION_STATIC||ownerMatches);
}
${edgeReceiverRecovery ? /* wgsl */ `fn dryPrepassUseExactReceiver(texel:vec2i,depth:f32,normal:vec3f,hit:DryHit)->bool{
  let geometry=textureLoad(dryPrepassGeometryTexture,texel,0);if(geometry.x<=0.0){return false;}
  if(!dryPrepassReceiverCompatible(textureLoad(dryPrepassIdentityTexture,texel,0).x,u32(round(geometry.w)),hit)){return false;}
  // Invert the old exp/pow >= .25 tests: ln(4)/24 and pow(.25,1/8).
  if(abs(geometry.x-depth)>0.057762265*max(depth,1e-3)){return false;}
  if(dot(normal,dryPrepassDecodeNormal(geometry.yz))<0.840896415){return false;}
  let packed=textureLoad(dryPrepassVisibilityKeyTexture,texel,0);if(all(packed.xy==DRY_PREPASS_INVALID_PACKED)){return false;}
  dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);dryPrepassState=1u;
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){dryPrepassGi=textureLoad(dryPrepassRadianceTexture,texel,0);dryPrepassGiState=1u;}
  if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,texel,0);dryPrepassRadianceState=1u;}
  return true;
}
fn dryPrepassRecoverExactReceiver(coordinate:vec2f,dims:vec2u,depth:f32,normal:vec3f,hit:DryHit)->bool{
  let nearest=vec2i(clamp(round(coordinate),vec2f(0.0),vec2f(dims)-vec2f(1.0)));
  // Two compact Chebyshev rings cover a one-full-pixel feature at every
  // supported reduced rate. The exceptional path has a hard 24-candidate cap.
  for(var radius=1;radius<=2;radius+=1){for(var y=-radius;y<=radius;y+=1){for(var x=-radius;x<=radius;x+=1){
    if(max(abs(x),abs(y))!=radius){continue;}
    let texel=clamp(nearest+vec2i(x,y),vec2i(0),vec2i(dims)-vec2i(1));
    if(dryPrepassUseExactReceiver(texel,depth,normal,hit)){return true;}
  }}}
  return false;
}
` : ""}
fn dryPrepassResolve(pixel:vec2f,depth:f32,normalIn:vec3f,hit:DryHit){
  ${voxelLightCache ? `if((dryVoxelLight.control.w&2u)!=0u){return;}` : ""}
  let dims=textureDimensions(dryPrepassGeometryTexture);
  let normal=normalize(normalIn);
  let coordinate=pixel*(vec2f(dims)/max(uniforms.viewport.xy,vec2f(1.0)))-vec2f(.5);
  let base=floor(coordinate);let fraction=coordinate-base;
  var accumulated0=vec4f(0.0);var accumulated1=vec4f(0.0);var accumulated2=vec4f(0.0);var weightSum=0.0;
  var accumulatedRadiance=vec4f(0.0);var radianceWeightSum=0.0;var accumulatedGi=vec4f(0.0);var giWeightSum=0.0;var bestRadianceWeight=0.0;var bestRadianceTexel=vec2i(0);var linearSafe=1u;
  for(var j=0u;j<2u;j+=1u){for(var i=0u;i<2u;i+=1u){
    let texel=vec2i(clamp(base+vec2f(f32(i),f32(j)),vec2f(0.0),vec2f(dims)-vec2f(1.0)));
    let geometry=textureLoad(dryPrepassGeometryTexture,texel,0);
    if(geometry.x<=0.0){linearSafe=0u;continue;}
    let bilinear=select(1.0-fraction.x,fraction.x,i==1u)*select(1.0-fraction.y,fraction.y,j==1u);
    let depthWeight=exp(-24.0*abs(geometry.x-depth)/max(depth,1e-3));
    let normalWeight=pow(max(dot(normal,dryPrepassDecodeNormal(geometry.yz)),0.0),8.0);
    let identityMatches=dryPrepassReceiverCompatible(textureLoad(dryPrepassIdentityTexture,texel,0).x,u32(round(geometry.w)),hit);
    if(!identityMatches){if(bilinear>1e-6){linearSafe=0u;}continue;}
    if(bilinear>1e-6&&(depthWeight<0.25||normalWeight<0.25)){linearSafe=0u;}
    let packed=textureLoad(dryPrepassVisibilityKeyTexture,texel,0);
    if(all(packed.xy==DRY_PREPASS_INVALID_PACKED)){linearSafe=0u;continue;}
    let guidedWeight=depthWeight*normalWeight;
    let weight=bilinear*select(guidedWeight,1.0,dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u);
    if(weight<=1e-6){continue;}
    accumulated0+=dryPrepassUnpack0(packed)*weight;
    accumulated1+=dryPrepassUnpack1(packed)*weight;
    accumulated2+=dryPrepassUnpack2(packed)*weight;
    if(weight>bestRadianceWeight){bestRadianceWeight=weight;bestRadianceTexel=texel;}
    if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u
      &&(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
        ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u)){accumulatedGi+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;giWeightSum+=weight;}
    if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u){accumulatedRadiance+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;radianceWeightSum+=weight;}
    weightSum+=weight;
  }}
  if(weightSum<${SVO_DRY_CONE_PREPASS_CONTRACT.minimumReconstructionWeight}){
    ${edgeReceiverRecovery ? `if(bestRadianceWeight>0.0&&dryPrepassUseExactReceiver(bestRadianceTexel,depth,normal,hit)){return;}
    if(dryPrepassRecoverExactReceiver(coordinate,dims,depth,normal,hit)){return;}` : ""}
    // A sub-prepass-pixel surface has no lawful screen-space receiver. Mark the
    // explicit exact edge tier and let the existing live cone closures run for
    // this pixel. Any unavailable live page still publishes its typed failure.
    dryPrepassExactEdgeState=1u;return;
  }
  dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;dryPrepassState=1u;
  if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
    ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){
    if(giWeightSum>=${SVO_DRY_CONE_PREPASS_CONTRACT.minimumReconstructionWeight}){dryPrepassGi=accumulatedGi/giWeightSum;dryPrepassGiState=1u;}
    else if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;}
  }
  if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["gated-linear"]}u&&linearSafe!=0u&&bestRadianceWeight>0.0){
    let reconstructed=textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0);if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}
  }else if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["joint-bilateral"]}u&&radianceWeightSum>1e-6){
    let reconstructed=accumulatedRadiance/radianceWeightSum;if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}
  }else if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u&&bestRadianceWeight>0.0){let reconstructed=textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0);if(reconstructed.a>=0.0){dryPrepassRadiance=reconstructed;dryPrepassRadianceState=1u;}}
}
` : "";
  const splitGlassKeyDeclarationWGSL = rasterGlassDiscovery
    ? /* wgsl */ `@group(${splitGroup}) @binding(6) var drySplitGlassKeyRead:texture_2d<u32>;
`
    : "";
  const splitRigidReadWGSL = /* wgsl */ `
fn drySplitGeometryAt(coordinate:vec2i)->vec4f{return textureLoad(drySplitGeometryRead,coordinate,0);}
fn drySplitIdentityAt(coordinate:vec2i)->vec4u{var value=textureLoad(drySplitOpaqueIdentityRead,coordinate,0);${experiments.surfaceMesh ? "value.x &= 0x8000ffffu;" : ""}return value;}
`;
  // One texel per pixel, written by the entry prepass and read by the primary
  // fragment alone. It is declared inside the split visibility group rather than
  // beside the lighting planes precisely so no other pipeline layout can reach
  // it: a secondary ray must never consult a plane built for primary rays.
  const primaryEntrySeedDeclarationWGSL = primaryEntrySeed ? /* wgsl */ `
@group(${splitGroup}) @binding(${SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.seedBinding}) var dryPrimaryEntrySeedRead:texture_2d<u32>;
` : "";
  const primaryWorkMapWGSL = primaryWorkMap ? /* wgsl */ `
@group(${splitGroup}) @binding(7) var dryPrimaryWorkMapWrite:texture_storage_2d<rgba32uint,write>;
var<private> dryPrimaryWorkNodeVisits:u32;
var<private> dryPrimaryWorkLeafVisits:u32;
var<private> dryPrimaryWorkVoxelCells:u32;
var<private> dryPrimaryWorkPlanarTests:u32;
var<private> dryPrimaryWorkCandidateNodes:u32;
var<private> dryPrimaryWorkPrimitiveTests:u32;
var<private> dryPrimaryWorkEntryState:u32;
fn dryPrimaryWorkReset(){
  dryPrimaryWorkNodeVisits=0u;dryPrimaryWorkLeafVisits=0u;dryPrimaryWorkVoxelCells=0u;
  dryPrimaryWorkPlanarTests=0u;dryPrimaryWorkCandidateNodes=0u;dryPrimaryWorkPrimitiveTests=0u;
  dryPrimaryWorkEntryState=0u;
}
fn dryPrimaryWorkPublish(coordinate:vec2i,hit:DryHit){
  let terminal=select(2u,1u,hit.t<DRY_MISS);
  let field=select(15u,hit.fieldSource&15u,hit.t<DRY_MISS);
  let nodeAndSeed=min(dryPrimaryWorkNodeVisits,65535u)|(min(dryPrimaryWorkEntryState,3u)<<16u);
  let leafAndResult=min(dryPrimaryWorkLeafVisits,65535u)|(terminal<<16u)|(field<<20u);
  let analytic=min(dryPrimaryWorkPlanarTests,255u)
    |(min(dryPrimaryWorkCandidateNodes,4095u)<<8u)
    |(min(dryPrimaryWorkPrimitiveTests,4095u)<<20u);
  textureStore(dryPrimaryWorkMapWrite,coordinate,
    vec4u(nodeAndSeed,leafAndResult,dryPrimaryWorkVoxelCells,analytic));
}
` : "";
  // The seed is a *per-ray* fact, not a per-pixel one. The primary fragment
  // loads it for the one ray it owns and `traceStaticFrom` takes it, which
  // clears it: every later ray from the same invocation — the walk behind a
  // thin dielectric, a refraction, anything reached through shading — starts
  // somewhere this plane was never asked about, and must fall back to the root.
  const primaryEntrySeedLibraryWGSL = primaryEntrySeed ? /* wgsl */ `
const DRY_PRIMARY_ENTRY_UNSEEDED:u32=0u;
// No voxel leaf covers this ray. Not "none was found" — the prepass draws every
// leaf the cursor could report, so an empty texel is a proof of absence and the
// exact planar seed is already the whole answer.
const DRY_PRIMARY_ENTRY_EMPTY:u32=1u;
const DRY_PRIMARY_ENTRY_LEAF:u32=2u;
var<private> dryPrimaryEntryState:u32;
var<private> dryPrimaryEntryMinimum:f32;
struct DryPrimaryEntrySeed{state:u32,minimum:f32}
fn dryPrimaryEntrySeedLoad(coordinate:vec2i){
  // The plane is only an answer on a frame that drew it. Withheld, the private
  // state stays at its zero-initialized UNSEEDED and every consumer below
  // no-ops, which is the root descent a build without the prepass compiles.
  if(dry.primaryEntry.x==0u){
    dryPrimaryEntryState=DRY_PRIMARY_ENTRY_UNSEEDED;dryPrimaryEntryMinimum=0.0;return;
  }
  let texel=textureLoad(dryPrimaryEntrySeedRead,coordinate,0);
  if(texel.y==${SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.emptyKey}u){
    dryPrimaryEntryState=DRY_PRIMARY_ENTRY_EMPTY;dryPrimaryEntryMinimum=0.0;return;
  }
  dryPrimaryEntryState=DRY_PRIMARY_ENTRY_LEAF;
  dryPrimaryEntryMinimum=max(bitcast<f32>(texel.x),0.0);
}
fn dryPrimaryEntrySeedTake(initialMinimum:f32)->DryPrimaryEntrySeed{
  // A caller that already starts past the origin is continuing some earlier
  // ray, so the plane is not about it even on the first take.
  let state=select(DRY_PRIMARY_ENTRY_UNSEEDED,dryPrimaryEntryState,initialMinimum<=0.0);
  dryPrimaryEntryState=DRY_PRIMARY_ENTRY_UNSEEDED;
  return DryPrimaryEntrySeed(state,dryPrimaryEntryMinimum);
}
` : "";
  const primaryEntrySeedTakeWGSL = primaryEntrySeed
    ? `let entrySeed=dryPrimaryEntrySeedTake(initialMinimum);${primaryWorkMap ? "dryPrimaryWorkEntryState=entrySeed.state;" : ""}` : "";
  // The one writer. Every other entry point in this module leaves the private
  // state at its zero-initialized `UNSEEDED`, which is why the seed plane is
  // reachable from the primary fragment and from nowhere else.
  const primaryEntrySeedLoadWGSL = primaryEntrySeed
    ? "dryPrimaryEntrySeedLoad(vec2i(input.position.xy));" : "";
  const primaryEntrySeedResolveWGSL = primaryEntrySeed ? /* wgsl */ `
  // Two of the three seed states retire the cursor outright.
  //
  // \`EMPTY\` says no voxel leaf covers this ray at all, so the planar seed is
  // the answer and there is nothing to descend for. \`LEAF\` with an entry at or
  // behind the seed says the same thing by distance: the return below keeps the
  // voxel only when it is strictly nearer, and every voxel on this ray lies at
  // or beyond \`entrySeed.minimum\`.
  if(entrySeed.state==DRY_PRIMARY_ENTRY_EMPTY){return seeded;}
  if(entrySeed.state==DRY_PRIMARY_ENTRY_LEAF&&!(entrySeed.minimum<seeded.t)){return seeded;}
` : "";
  const primaryEntrySeedMinimumWGSL = primaryEntrySeed ? /* wgsl */ `
  // Everything nearer than the first voxel leaf is empty space by construction,
  // so the cursor's near bound moves there. \`svoRayAabbWithInverse\` seeds
  // \`enter\` from \`ray.tMin\`, so every child box the ray meets only in front of
  // it fails its interval test instead of being expanded, and the reported
  // \`tEnter\` of every surviving leaf is unchanged — the recorded entry is a
  // lower bound on all of them.
  if(entrySeed.state==DRY_PRIMARY_ENTRY_LEAF){minimum=max(minimum,entrySeed.minimum);}
` : "";
  const splitDeclarationsWGSL = split ? /* wgsl */ `// Split visibility/lighting bridge. The visibility entry writes exact primary
// geometry while the lighting entry reads it together with the final G-buffer.
// Separate pipelines expose only the bindings reachable from their entry point.
@group(${splitGroup}) @binding(0) var drySplitGeometryWrite:texture_storage_2d<rgba32float,write>;
@group(${splitGroup}) @binding(1) var drySplitGeometryRead:texture_2d<f32>;
@group(${splitGroup}) @binding(4) var drySplitOpaqueIdentityWrite:texture_storage_2d<rg32uint,write>;
@group(${splitGroup}) @binding(5) var drySplitOpaqueIdentityRead:texture_2d<u32>;
${splitGlassKeyDeclarationWGSL}${primaryEntrySeedDeclarationWGSL}${primaryWorkMapWGSL}
${splitRigidReadWGSL}
` : "";
  const voxelLightCacheGroup = splitGroup + 1;
  const voxelLightCacheWGSL = voxelLightCache ? /* wgsl */ `
// Persistent world-space visibility for directional light slot zero. The
// request bitset is frame-local; the atlas texels survive camera motion and
// carry a 16-bit lighting epoch plus the representative surface normal.
struct DryVoxelLightCacheParams{control:vec4u,atlasPages:vec4u}
struct DryVoxelLightAddress{atlas:vec3u,voxelIndex:u32,valid:u32}
struct DryVoxelLightQueue{
  distinct:atomic<u32>,misses:atomic<u32>,hits:atomic<u32>,queued:atomic<u32>,
  populated:atomic<u32>,rejected:atomic<u32>,overflow:atomic<u32>,reserved:atomic<u32>,
  entries:array<vec2u>,
}
@group(${voxelLightCacheGroup}) @binding(0) var dryVoxelLightCacheRead:texture_3d<u32>;
@group(${voxelLightCacheGroup}) @binding(1) var<uniform> dryVoxelLight:DryVoxelLightCacheParams;
@group(${voxelLightCacheGroup}) @binding(2) var dryVoxelLightCacheWrite:texture_storage_3d<rg32uint,write>;
@group(${voxelLightCacheGroup}) @binding(3) var<storage,read_write> dryVoxelLightRequests:array<atomic<u32>>;
@group(${voxelLightCacheGroup}) @binding(4) var<storage,read_write> dryVoxelLightQueue:DryVoxelLightQueue;
@group(${voxelLightCacheGroup}) @binding(5) var<storage,read> dryVoxelLightPages:array<vec4u>;
${reduced ? "" : "var<private> dryCurrentLightSlot:u32;"}
var<private> dryVoxelLightConsumerEligible:u32;
fn dryVoxelLightPackNormal(normalIn:vec3f)->u32{
  let normal=normalize(normalIn);var oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
  if(normal.z<0.0){oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0.0));}
  let encoded=vec2u(round(clamp(oct*.5+.5,vec2f(0.0),vec2f(1.0))*255.0));return encoded.x|(encoded.y<<8u);
}
fn dryVoxelLightUnpackNormal(packed:u32)->vec3f{
  let oct=vec2f(f32(packed&255u),f32((packed>>8u)&255u))/255.0*2.0-1.0;
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));if(normal.z<0.0){let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));normal=vec3f(folded,normal.z);}return normalize(normal);
}
// The cache is one entry per texel of a *base* page, and the base is the
// opacity floor — the pyramid's finest resident level, not level zero. Under a
// floor of one its entries cover 2^3 finest voxels each, which is the same
// coarsening the opacity field it caches visibility against already took. At
// floor zero every line below is the arithmetic that shipped.
fn dryVoxelLightAddress(position:vec3f)->DryVoxelLightAddress{
  if((dryVoxelLight.control.w&1u)==0u||dry.nodeMipDirect.w==0u){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let baseLevel=dryNodeMipOpacityLevelFloor();
  let virtualVoxelFloor=floor((position-dry.nodeMipOrigin.xyz)/max(dry.mapping.cellSize*exp2(f32(baseLevel)),vec3f(1e-6)));
  if(any(virtualVoxelFloor<vec3f(0.0))){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let voxel=vec3u(virtualVoxelFloor);let pageCoordinate=voxel/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let pageIndex=dryNodeMipFind(baseLevel,pageCoordinate);
  if(pageIndex==0xffffffffu||pageIndex>=dryVoxelLight.control.x||!dryNodeMipPageValid(pageIndex)){return DryVoxelLightAddress(vec3u(0u),0u,0u);}
  let local=voxel-pageCoordinate*${SVO_NODE_MIP_LAYOUT.interiorSize}u;
  let localIndex=local.x+local.y*${SVO_NODE_MIP_LAYOUT.interiorSize}u+local.z*${SVO_NODE_MIP_LAYOUT.interiorSize ** 2}u;
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));
  let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  return DryVoxelLightAddress(atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local,pageIndex*${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u+localIndex,1u);
}
fn dryVoxelLightVisibility(position:vec3f,normal:vec3f)->vec2f{
  if(dryCurrentLightSlot!=0u||dryVoxelLightConsumerEligible==0u){return vec2f(0.0);}
  let address=dryVoxelLightAddress(position);if(address.valid==0u){return vec2f(0.0);}
  let encoded=textureLoad(dryVoxelLightCacheRead,vec3i(address.atlas),0).xy;
  if(encoded.x==0u||(encoded.y>>16u)!=(dryVoxelLight.control.z&0xffffu)){return vec2f(0.0);}
  if(dot(normalize(normal),dryVoxelLightUnpackNormal(encoded.y&0xffffu))<.9){return vec2f(0.0);}
  return vec2f(f32(encoded.x-1u)/65534.0,1.0);
}
@compute @workgroup_size(8,8) fn dryVoxelLightDemandMain(@builtin(global_invocation_id) id:vec3u){
  let dimensions=textureDimensions(drySplitGeometryRead);if(any(id.xy>=dimensions)){return;}
  let coordinate=vec2i(id.xy);let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return;}
  let metadata=drySplitIdentityAt(coordinate).y;let motionKind=(metadata>>24u)&3u;let feature=(metadata>>16u)&15u;
  if(motionKind!=DRY_GBUFFER_MOTION_STATIC){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  let uv=vec2f((f32(id.x)+.5)/f32(dimensions.x),1.0-(f32(id.y)+.5)/f32(dimensions.y));let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());let position=ro+rd*geometry.w;
  let address=dryVoxelLightAddress(position);if(address.valid==0u){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  let word=address.voxelIndex>>5u;let bit=1u<<(address.voxelIndex&31u);let previous=atomicOr(&dryVoxelLightRequests[word],bit);if((previous&bit)!=0u){return;}
  atomicAdd(&dryVoxelLightQueue.distinct,1u);let packedNormal=dryVoxelLightPackNormal(geometry.xyz);
  let encoded=textureLoad(dryVoxelLightCacheRead,vec3i(address.atlas),0).xy;
  if(encoded.x==0u&&(encoded.y>>16u)==(dryVoxelLight.control.z&0xffffu)&&(encoded.y&0xffffu)==0xffffu){atomicAdd(&dryVoxelLightQueue.rejected,1u);return;}
  if(encoded.x!=0u&&(encoded.y>>16u)==(dryVoxelLight.control.z&0xffffu)){
    if(dot(normalize(geometry.xyz),dryVoxelLightUnpackNormal(encoded.y&0xffffu))>=.9){atomicAdd(&dryVoxelLightQueue.hits,1u);}else{atomicAdd(&dryVoxelLightQueue.rejected,1u);}return;
  }
  atomicAdd(&dryVoxelLightQueue.misses,1u);let queueIndex=atomicAdd(&dryVoxelLightQueue.queued,1u);
  if(queueIndex<dryVoxelLight.control.y){dryVoxelLightQueue.entries[queueIndex]=vec2u(address.voxelIndex,packedNormal);}else{atomicAdd(&dryVoxelLightQueue.overflow,1u);}
}
fn dryVoxelLightReject(pageIndex:u32,local:vec3u){
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  let atlas=atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local;
  textureStore(dryVoxelLightCacheWrite,vec3i(atlas),vec4u(0u,((dryVoxelLight.control.z&0xffffu)<<16u)|0xffffu,0u,0u));atomicAdd(&dryVoxelLightQueue.rejected,1u);
}
@compute @workgroup_size(64) fn dryVoxelLightPopulateMain(@builtin(global_invocation_id) id:vec3u){
  let count=min(atomicLoad(&dryVoxelLightQueue.queued),dryVoxelLight.control.y);if(id.x>=count){return;}
  let entry=dryVoxelLightQueue.entries[id.x];let pageIndex=entry.x/${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u;
  let localIndex=entry.x-pageIndex*${SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.voxelsPerPage}u;if(pageIndex>=dryVoxelLight.control.x){return;}
  let local=vec3u(localIndex%${SVO_NODE_MIP_LAYOUT.interiorSize}u,(localIndex/${SVO_NODE_MIP_LAYOUT.interiorSize}u)%${SVO_NODE_MIP_LAYOUT.interiorSize}u,localIndex/${SVO_NODE_MIP_LAYOUT.interiorSize ** 2}u);
  // Only base-level pages carry cache entries, and the base is the opacity
  // floor; a coarser page's texels are not the lattice the addresses above use.
  let baseLevel=dryNodeMipOpacityLevelFloor();
  let page=dryVoxelLightPages[pageIndex];if(page.w!=baseLevel){return;}let worldVoxel=page.xyz*${SVO_NODE_MIP_LAYOUT.interiorSize}u+local;
  let position=dry.nodeMipOrigin.xyz+(vec3f(worldVoxel)+vec3f(.5))*dry.mapping.cellSize*exp2(f32(baseLevel));let normal=dryVoxelLightUnpackNormal(entry.y);
  var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);let coverage=dryNodeMipAt(position,0.0,&pageCache);
  if(coverage.valid==0u||(coverage.sample.solidMaximum>.85&&coverage.sample.solidMean<.08)){dryVoxelLightReject(pageIndex,local);return;}
  let light=dryLighting.lights[0];if(dryLighting.metadata.x==0u||light.identity.x!=SVO_LIGHT_DIRECTIONAL||light.identity.w!=dryLighting.metadata.y){return;}
  let sample=dryLightSample(light,0u,position);var visibility=0.0;
  if(sample.valid!=0u&&dot(normal,sample.towardLight)>0.0){let maximumDistance=directionalLightSceneExitDistance(position,sample.towardLight);let ray=dryBiasedVisibilityRayUnit(position,normal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);let cell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let escape=cell*dry.tuningRays1.z;let coneMaximum=max(0.0,ray.tMax_m-escape*dot(normal,sample.towardLight));let cone=dryConeVisibility(ray.origin_m+normal*escape,sample.towardLight,dry.tuningRays1.y,coneMaximum,normal,false);if(cone.valid==0u){dryVoxelLightReject(pageIndex,local);return;}visibility=cone.transmittance;}
  let atlasPages=max(dryVoxelLight.atlasPages.xyz,vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));
  let atlas=atlasPage*${SVO_NODE_MIP_LAYOUT.physicalSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.apron}u)+local;let packed=u32(round(clamp(visibility,0.0,1.0)*65534.0))+1u;
  textureStore(dryVoxelLightCacheWrite,vec3i(atlas),vec4u(packed,((dryVoxelLight.control.z&0xffffu)<<16u)|(entry.y&0xffffu),0u,0u));atomicAdd(&dryVoxelLightQueue.populated,1u);
}
` : "";
  const voxelLightCacheShortcutWGSL = voxelLightCache ? /* wgsl */ `let cachedVoxelVisibility=dryVoxelLightVisibility(position,geometricNormal);if(cachedVoxelVisibility.y>0.0){let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);let raw=select(cachedVoxelVisibility.x,0.0,rigidBlocker.t<ray.tMax_m);return vec3f(mix(1.0,raw,dry.tuningRays0.y));}if(dryCurrentLightSlot==0u&&(dryVoxelLight.control.w&2u)!=0u){dryCurrentLightSlot=0xffffffffu;}` : "";
  const prepassResolveCallWGSL = reduced
    ? /* wgsl */ `dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);dryPrepassState=0u;dryPrepassRadianceState=0u;dryPrepassGiState=0u;dryPrepassExactEdgeState=0u;dryCurrentLightSlot=0xffffffffu;if(opaque.t<DRY_MISS&&!dryHitThinDielectric(opaque)&&(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u){if(dryNodeMipReady()){dryPrepassResolve(input.position.xy,opaque.t,opaque.normal,opaque);}else{dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;}}${voxelLightCache ? "if((dryVoxelLight.control.w&2u)!=0u){dryPrepassRadianceState=0u;dryPrepassGiState=0u;}" : ""}`
    : "";
  const prepassShadowShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u&&dryCurrentLightSlot<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u){let prepassRigidBlocked=anyBodyBlockerIgnoring(ray.origin_m,towardLight,ownerId,ray.tMax_m);let raw=select(dryPrepassChannel(1u+dryCurrentLightSlot),0.0,prepassRigidBlocked);return vec3f(mix(1.0,raw,dry.tuningRays0.y));}`
    : "";
  const prepassContactShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassState==1u){let prepassRadius=dryContactVisibilityRadius();if(prepassRadius<=0.0){return vec3f(1.0);}let prepassCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let prepassOrigin=position+normalize(geometricNormal)*prepassCell*.2;let prepassSamples=max(dry.tuningCounts1.z,dry.tuningCounts1.y);var prepassUnblocked=0.0;for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=prepassSamples){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let prepassRigidBlocked=anyBodyBlockerIgnoring(prepassOrigin,rotated,ownerId,prepassRadius);prepassUnblocked+=select(1.0,0.0,prepassRigidBlocked);}let raw=clamp(dryPrepassData0.x*(prepassUnblocked/f32(prepassSamples)),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));}`
    : "";
  const prepassBodyBlockerWGSL = reduced ? /* wgsl */ `fn anyBodyBlockerIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32,tMax:f32)->bool {
  // Every shaded pixel calls this once per light and once per contact sample, so
  // at 1500x1500 the body loop runs tens of millions of times a frame and its
  // per-iteration read of bodies[] dominates. One sphere enclosing the whole set
  // rejects the overwhelming majority of those rays before a single body is
  // fetched: rigid bodies occupy a small part of a scene, and contact rays are
  // short. The bound is published per frame, so it costs nothing to consult.
  if(!svoRigidBoundsIntersect(ro,rd,tMax)){return false;}
  for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,tMax)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,tMax)){continue;}if(bodyHit(ro,rd,body).t<tMax){return true;}}
  return false;
}
` : "";
  const prepassLightSlotWGSL = reduced || voxelLightCache ? /* wgsl */ `dryCurrentLightSlot=lightIndex;` : "";
  const prepassRadianceShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassRadianceState==1u&&hit.motionKind==DRY_GBUFFER_MOTION_STATIC){return max(dryPrepassRadiance.rgb,vec3f(0.0));}`
    : "";
  const prepassGiShortcutWGSL = reduced
    ? /* wgsl */ `if(dryPrepassGiState==1u){if(dryPrepassGi.a<0.0){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0),1u);}`
    : "";
  const splitVisibilityGlassDiscoveryWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(opaque.materialId,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`
    : /* wgsl */ `let glass=traceGlass(ro,rd,0.0,opaque.t);let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;let coordinate=vec2i(input.position.xy);textureStore(drySplitGeometryWrite,coordinate,vec4f(opaque.normal,opaque.t));let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u);let glassKey=select(0u,glass.recordIndex+1u,glassVisible);let packedOpaqueMaterial=(opaque.materialId&0x8000ffffu)|((glassKey&0x1ffu)<<16u);textureStore(drySplitOpaqueIdentityWrite,coordinate,vec4u(packedOpaqueMaterial,opaqueMetadata,0u,0u));let generation=dryPublicationGeneration();`;
  const splitVisibilityGlassReturnWGSL = rasterGlassDiscovery
    ? ""
    : /* wgsl */ `if(glassVisible){let record=dryGlassPane(glass.recordIndex);let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);let targets=svoGBufferSurface(vec3f(0.0),glass.hit.t_m,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_GLASS),SVO_FEATURE_SMOOTH);return drySplitVisibilityOut(targets,dryHardwareDepth(glass.hit.t_m,rd,forward));}`;
  const splitOpaqueMaterialDecodeWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let opaqueMaterial=packedOpaqueMaterial;`
    : /* wgsl */ `let opaqueMaterial=select(packedOpaqueMaterial&0xffffu,0x80000000u|(packedOpaqueMaterial&0xffffu),(packedOpaqueMaterial&0x80000000u)!=0u);`;
  const splitGlassKeyLoadWGSL = rasterGlassDiscovery
    ? /* wgsl */ `let glassKey=textureLoad(drySplitGlassKeyRead,coordinate,0).x;`
    : /* wgsl */ `let glassKey=(packedOpaqueMaterial>>16u)&0x1ffu;`;
  const splitPrimaryTraceWGSL = rasterRigidDiscovery ? "traceStatic(ro,rd)" : "traceOpaqueScene(ro,rd)";
  const rasterPrimaryEntryWGSL = rasterPrimary ? /* wgsl */ `
// Raster-assisted primary visibility. Every plane the deferred lighting pass
// consumes is a depth-tested colour attachment here, so no pass in this graph
// writes primary geometry through an untested storage texture.
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding}) var<storage,read> svoBrickInstances:array<SvoBrickInstance>;
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.coverageCountBinding}) var<storage,read_write> svoBrickCoverageCounts:array<atomic<u32>>;
@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding}) var<storage,read_write> svoBrickCoverageCandidates:array<u32>;
${screenSpaceTerminationPixels > 0 ? `@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.primaryGeometryBinding}) var dryRasterPrimaryGeometryRead:texture_2d<f32>;` : ""}
${screenSpaceTerminationPixels > 0 ? `@group(${splitGroup}) @binding(${SVO_BRICK_RASTER_CONTRACT.primaryIdentityBinding}) var dryRasterPrimaryIdentityRead:texture_2d<u32>;` : ""}
${svoScenePrimitiveBandReadWGSL(splitGroup + 1, SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel)}
${svoRasterCoverageOverflowSignalWGSL}
// Pixels the count buffer actually addresses.
//
// The allocation now carries an eight-word tail past the pixel range — the
// overflow flag and the indirect draw args — so arrayLength stopped being a
// pixel bound. It is still a *safe* one (every guard using it is permissive by
// exactly the tail), but a fragment that read the tail as a pixel would corrupt
// the draw the overflow pass is about to issue, so the bound is explicit.
fn dryCoveragePixelLimit()->u32{
  let words=arrayLength(&svoBrickCoverageCounts);
  if(words<=SVO_RASTER_COVERAGE_TAIL_WORDS){return 0u;}
  return min(words-SVO_RASTER_COVERAGE_TAIL_WORDS,
    max(u32(uniforms.viewport.x),1u)*max(u32(uniforms.viewport.y),1u));
}
${svoBrickRasterSharedWGSL}
struct DryRasterPrimaryOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,
  @builtin(frag_depth) hardwareDepth:f32,
}
struct SvoBrickRasterVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) proxyMinimum:vec3f,
  @location(1) @interpolate(flat) proxyMaximum:vec3f,
  @location(2) @interpolate(flat) nodeIndex:u32,
  @location(3) @interpolate(flat) voxelOffset:u32,
  @location(4) @interpolate(flat) instanceIndex:u32,
}
fn dryRasterPrimaryReset(){dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;dryThickGlassFailure=0u;}
fn dryRasterPrimaryCamera()->mat4x3f{
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));return mat4x3f(ro,forward,right,normalize(cross(right,forward)));
}
fn dryRasterPrimaryRay(pixel:vec2f,camera:mat4x3f)->vec3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  return normalize(camera[1]+camera[2]*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+camera[3]*ndc.y*cameraTanHalfFov());
}
// A producer whose surface is one shape and whose shading is another states
// both: \`geometricNormal\` is the face that was actually rasterised and
// \`opaque.normal\` is what the pixel shades with. The geometric slot of the
// published G-buffer takes the former, the f32 geometry plane the latter (so
// shading keeps full precision), and the identity plane carries the face for
// the deferred lighting's ray bias. Only the voxel mesh's filtered detail has
// two; everything else calls \`dryRasterPrimarySurface\`, which passes the one
// normal twice and is byte for byte what it always was.
fn dryRasterPrimarySurface(opaque:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32)->DryRasterPrimaryOut{
  return dryRasterPrimaryFacedSurface(opaque,opaque.normal,ro,rd,forward,producer);
}
fn dryRasterPrimaryFacedSurface(opaque:DryHit,geometricNormal:vec3f,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32)->DryRasterPrimaryOut{
  let generation=dryPublicationGeneration();
  let media=dryMediumPair(rd,opaque.normal,DRY_MEDIUM_OPAQUE);
  let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);
  let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(producer);
  if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}
  let targets=svoGBufferSurface(vec3f(0.0),opaque.t,geometricNormal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
  let opaqueMetadata=(opaque.ownerId&0xffffu)|((opaque.featureId&15u)<<16u)|((opaque.fieldSource&15u)<<20u)|((opaque.motionKind&3u)<<24u)|((opaque.motionValid&1u)<<26u)|dryOpaqueFaceWord(geometricNormal,opaque.normal);
  return DryRasterPrimaryOut(targets.packedSurface,targets.identityMedia,vec4f(opaque.normal,opaque.t),vec2u(opaque.materialId,opaqueMetadata),dryHardwareDepth(opaque.t,rd,forward));
}
fn dryRasterPrimaryMiss()->DryRasterPrimaryOut{
  let targets=svoGBufferMiss(vec3f(0.0),0u,dryPublicationGeneration(),DRY_GBUFFER_NO_INTERSECTION,
    svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND));
  return DryRasterPrimaryOut(targets.packedSurface,targets.identityMedia,vec4f(0.0,1.0,0.0,DRY_MISS),vec2u(0u,0u),0.0);
}
// The brick payload stores one owner per voxel. That is sufficient to find a
// nearby surface, but it cannot represent a sub-voxel visibility boundary
// between two projected primitives. Raster one conservative box per authored
// finite primitive and repeat the shared exact intersection in the fragment;
// reversed-Z then chooses the true nearest surface across all owners.
struct DryScenePrimitiveVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) primitiveIndex:u32,
}
// The ray's span through the primitive's own oriented box, as the march bracket.
//
// primitiveHit brackets a marched kind by its *bounding sphere*, and the ABI
// states the assumption that makes 48 iterations sufficient: "callers hand in a
// bounded interval - one traversed voxel in the renderer - so the march starts
// within a cell of the surface". This pass was handing in 0 .. DRY_MISS, so
// that assumption did not hold on the path carrying 78% of the frame. For a
// round cone the bounding sphere is halfLength + radius, many times the
// thickness the ray actually crosses, and every step of that excess is a full
// SDF evaluation.
//
// The local box is the same containment the raster proxy already relies on, so
// a hit outside this span does not exist. Returns x > y when the ray misses the
// box, which is a hit the exact test cannot produce either.
//
// The span is padded because containment is exact in real arithmetic and not in
// floating point: the exact solve puts a grazing cone hit up to 1.3e-5 *before*
// its own box entry (tests/svo-scene-primitive-march-span.test.ts finds it on
// the authored hero set), and an unpadded bracket would drop that sliver of
// silhouette on the one path carrying most of the frame. A grazing ray converts
// a tiny spatial error into a much larger one in t, so the pad is relative to
// the span rather than absolute. Widening the bracket cannot introduce a hit —
// the exact test still decides — it only declines to save a step or two.
fn dryScenePrimitiveMarchSpan(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f)->vec2f{
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let orientationLength=length(record.orientation);
  if(any(localExtent<vec3f(0.0))||!(orientationLength>1e-8)){return vec2f(0.0,DRY_MISS);}
  let inverse=vec4f(-record.orientation.xyz,record.orientation.w)/orientationLength;
  let localOrigin=svoQuaternionRotate(inverse,ro-svoPrimitiveCenter_m(record));
  let localDirection=svoQuaternionRotate(inverse,rd);
  let span=svoRayAabbWithInverse(SvoRay(localOrigin,0.0,localDirection,DRY_MISS),
    1.0/localDirection,mat2x3f(-localExtent,localExtent));
  if(span.x==0.0){return vec2f(DRY_MISS,0.0);}
  let pad=max(${SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M},${SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE}*span.z);
  return vec2f(max(0.0,span.y-pad),span.z+pad);
}
fn dryScenePrimitiveWorldExtent(localExtent:vec3f,orientation:vec4f)->vec3f{
  let q=orientation/length(orientation);
  return abs(svoQuaternionRotate(q,vec3f(localExtent.x,0.0,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,localExtent.y,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,0.0,localExtent.z)))+vec3f(1e-5);
}
// Conservative eight-corner frustum rejection for a proxy box.
//
// Transcribed from svoBrickFrustumVisible (webgpu-svo-brick-raster.ts), which
// is the repo's one proven form of this test: signed lateral from the view
// depth so corners behind the camera fold the half-space the right way, no far
// plane, and a box is rejected only when all eight corners fall outside the
// *same* plane. That is conservative in the only direction that matters — it
// can decline to cull, never cull something a fragment could have used.
//
// It takes the basis this vertex stage has already built rather than rebuilding
// one, because the caller has it in hand and the two must not be able to drift.
//
// Fails open. dryRasterPrimaryCamera() has no guard against a degenerate
// cameraTarget - cameraPosition, and none against a camera looking straight
// down the world up axis, either of which makes a basis vector NaN. A NaN
// comparison is false, so outside would clear itself and the box would draw —
// but that is an accident of IEEE rather than a contract, so the degenerate
// basis is rejected explicitly and the record is drawn.
fn dryScenePrimitiveFrustumVisible(camera:mat4x3f,aspect:f32,bounds:mat2x3f)->bool{
  let basis=camera[1]+camera[2]+camera[3];
  if(!(dot(basis,basis)<3.402823e38)){return true;}
  let tanHalfFov=cameraTanHalfFov();var outside=vec4<bool>(true,true,true,true);var outsideNear=true;
  for(var corner=0u;corner<8u;corner+=1u){
    let point=vec3f(select(bounds[0].x,bounds[1].x,(corner&1u)!=0u),select(bounds[0].y,bounds[1].y,(corner&2u)!=0u),select(bounds[0].z,bounds[1].z,(corner&4u)!=0u));
    let relative=point-camera[0];let viewDepth=dot(relative,camera[1]);let lateral=viewDepth*tanHalfFov;
    let x=dot(relative,camera[2]);let y=dot(relative,camera[3]);
    outsideNear=outsideNear&&(viewDepth<DRY_REVERSED_Z_NEAR_M);
    outside=vec4<bool>(outside.x&&(x+lateral*aspect<0.0),outside.y&&(lateral*aspect-x<0.0),
      outside.z&&(y+lateral<0.0),outside.w&&(lateral-y<0.0));
  }
  return !(outsideNear||any(outside));
}
// Tightening this proxy to the primitive's own oriented box was measured and
// does not pay, which is worth stating because the coverage argument for it is
// correct and still tempting: a hose spelled as 336 diagonal round cones has an
// axis-aligned box several times the shape, and the oriented box halves mean
// covering proxies per pixel over the hero framing (16.1 to 7.7, median 15 to
// 3). Interleaved and paired at 1600x1240 that bought -5.2, +3.7 and -24.1 ms
// against a control arm whose own spread was 356.7 to 385.8 — no effect, and
// the sign is not even stable. Fragment *count* is not what this pass is bound
// by; per-fragment march cost is, and the proxies the oriented box shrinks are
// the cheap round cones rather than the tapered-sweep clusters beside them.
fn dryScenePrimitiveProxyVertex(vertexIndex:u32,primitiveIndex:u32)->DryScenePrimitiveVertexOut{
  var position=vec4f(2.0,2.0,0.0,1.0);
  if(primitiveIndex>=dry.metadata.x){return DryScenePrimitiveVertexOut(position,primitiveIndex);}
  let record=dryPrimitive(primitiveIndex);let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let orientationLength=length(record.orientation);
  if(any(localExtent<vec3f(0.0))||!(orientationLength>1e-8)){return DryScenePrimitiveVertexOut(position,primitiveIndex);}
  let extent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);let minimum=centre-extent;let maximum=centre+extent;
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let forward=camera[1];let right=camera[2];let up=camera[3];let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let margin=vec3f(${4 * SVO_DRY_SCENE_REVERSED_Z_NEAR_M});
  if(all(ro>=minimum-margin)&&all(ro<=maximum+margin)){
    var screen=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(-1.0,3.0),vec2f(3.0,-1.0));
    if(vertexIndex<3u){position=vec4f(screen[vertexIndex],1.0,1.0);}
  }else{
    // ~501 authored records emitted 36 vertices each, every frame, with no view
    // test of any kind. A box the frustum does not touch cannot produce a
    // fragment, so this only ever removes work: rejection collapses all
    // thirty-six vertices onto the same degenerate clip point, which is this
    // file's established "drop this record" idiom and the same one the near-field
    // band uses. The eye-inside branch above is tested first and is never
    // reachable from here — a box containing the eye covers every pixel, and its
    // back faces would clip against the near plane rather than fill the screen.
    if(!dryScenePrimitiveFrustumVisible(camera,aspect,mat2x3f(minimum,maximum))){
      return DryScenePrimitiveVertexOut(vec4f(2.0,2.0,0.0,1.0),primitiveIndex);
    }
    let world=mix(minimum,maximum,svoBrickBoxCorner(vertexIndex));let relative=world-ro;let viewDepth=dot(relative,forward);
    position=vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,viewDepth);
  }
  return DryScenePrimitiveVertexOut(position,primitiveIndex);
}
// The exact authored set, minus only what the frustum cannot see. This entry
// point stays band-free on purpose — it feeds the direct fragment, which is the
// overflow fallback and the control the coverage arm is measured against, and a
// control that had already had records removed from it would measure nothing.
// The frustum rejection inside the shared proxy body is not that kind of
// removal: it applies identically to all three arms, so the set agreement below
// holds, and it drops only records no pixel of any arm could have marched.
@vertex fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.vertex}(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) primitiveIndex:u32,
)->DryScenePrimitiveVertexOut{
  return dryScenePrimitiveProxyVertex(vertexIndex,primitiveIndex);
}
// The banded set, for the coverage and overflow passes.
//
// Rejection happens here rather than in the fragment because this is where it is
// free: all thirty-six vertices collapse onto the same clip-space point, the
// triangles are degenerate, and the rasterizer produces nothing at all — so a
// record the band dropped costs no fragment, no arena slot, and no march. The
// two passes must agree on the set (overflow re-runs the pixels whose coverage
// list was full, and a list built from a different set would be re-marched
// against the wrong geometry), which is exactly why they share this entry point.
@vertex fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.bandVertex}(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) primitiveIndex:u32,
)->DryScenePrimitiveVertexOut{
  if(!dryScenePrimitiveBandMember(primitiveIndex)){
    return DryScenePrimitiveVertexOut(vec4f(2.0,2.0,0.0,1.0),primitiveIndex);
  }
  return dryScenePrimitiveProxyVertex(vertexIndex,primitiveIndex);
}
// Scene-primitive fragments carry their own output struct for the same reason
// the brick ones do: the hidden-surface-removal probe drops frag_depth here
// without disturbing any other pass sharing DryRasterPrimaryOut.
struct DrySceneRasterOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,${scenePrimitiveHsrProbe ? "" : `
  @builtin(frag_depth) hardwareDepth:f32,`}
}
fn drySceneRasterOut(surface:DryRasterPrimaryOut)->DrySceneRasterOut{
  return DrySceneRasterOut(surface.packedSurface,surface.identityMedia,surface.geometry,surface.opaqueIdentity${scenePrimitiveHsrProbe ? "" : ",surface.hardwareDepth"});
}
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.fragment}(input:DryScenePrimitiveVertexOut)->DrySceneRasterOut{
  dryRasterPrimaryReset();
${scenePrimitiveHsrProbe
    ? `  var probe=dryRasterPrimaryMiss();
  if(input.primitiveIndex<dry.metadata.x){
    let record=dryPrimitive(input.primitiveIndex);
    if(!dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){
      let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
      ${scenePrimitiveMarchSpanWGSL}
      if(exact.t<DRY_MISS){probe=dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);}
    }
  }
  return drySceneRasterOut(probe);`
    : `  if(input.primitiveIndex>=dry.metadata.x){discard;}
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){discard;}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  ${scenePrimitiveMarchSpanWGSL}
  if(!(exact.t<DRY_MISS)){discard;}
  return drySceneRasterOut(dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE));`}
}
// Background. It used to be background *and terrain*, and the ground it drew is
// voxels now, so what is left is the clear: every pixel no brick instance covers
// publishes an explicit miss into the G-buffer rather than whatever the previous
// frame left there. The megakernel's octree stack, rigid loop and pane loop are
// all absent here, which is the register budget this mode buys back.
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.background}(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  return dryRasterPrimaryMiss();
}
@vertex fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.vertex}(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) instanceIndex:u32)->SvoBrickRasterVertexOut{
  let record=svoBrickInstances[instanceIndex];
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  var position=vec4f(0.0,0.0,0.0,1.0);
  // Back faces let a camera inside a brick still shade it, but they clip once
  // the box reaches the near plane. Those few instances cover the screen at the
  // nearest depth instead; the fragment's own box intersection rejects the
  // pixels their proxy does not actually contain.
  let margin=vec3f(${4 * SVO_DRY_SCENE_REVERSED_Z_NEAR_M});
  if(all(ro>=record.proxyMinimum-margin)&&all(ro<=record.proxyMaximum+margin)){
    var screen=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(-1.0,3.0),vec2f(3.0,-1.0));
    if(vertexIndex<3u){position=vec4f(screen[vertexIndex],1.0,1.0);}
  }else{
    let world=mix(record.proxyMinimum,record.proxyMaximum,svoBrickBoxCorner(vertexIndex));
    let relative=world-ro;let viewDepth=dot(relative,forward);
    // Constant clip-space z with w = view depth is exactly the reversed-Z
    // infinite-far projection: the interpolated depth is near/viewDepth.
    position=vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,viewDepth);
  }
  return SvoBrickRasterVertexOut(position,record.proxyMinimum,record.proxyMaximum,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,record.voxelOffset,instanceIndex);
}
// Stage one is deliberately coverage-only. Ordinary fragments append their
// candidate and stop immediately; only overlaps beyond the fixed arena write
// the throwaway colour. No path traces a payload or writes fragment depth. The
// expensive resolve below consequently runs exactly once per pixel rather than
// once per overlapping brick.
${brickCoverageOutputWGSL}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.coverage}(
  input:SvoBrickRasterVertexOut,
 )${brickCoverageReturnTypeWGSL}{
  let width=max(u32(uniforms.viewport.x),1u);let coordinate=vec2u(input.position.xy);
  let pixel=coordinate.y*width+coordinate.x;
  if(pixel>=dryCoveragePixelLimit()){${brickCoverageMissWGSL}}
  ${screenSpaceBrickCoverageWGSL}
  let slot=atomicAdd(&svoBrickCoverageCounts[pixel],1u);
  if(slot<${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){
    let address=pixel*${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u+slot;
    if(address<arrayLength(&svoBrickCoverageCandidates)){svoBrickCoverageCandidates[address]=input.instanceIndex;}
    discard;
  }
  // The fragment that takes the *first* slot past capacity raises the flag, so
  // the counter counts overflowing pixels rather than surplus fragments — and
  // so a frame with none of them can skip the overflow draw entirely.
  if(slot==${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){svoRasterCoverageOverflowSignal();}
  ${brickCoverageOverflowWGSL}
}
fn dryBrickCoveragePixel(position:vec2f)->u32{
  return u32(position.y)*max(u32(uniforms.viewport.x),1u)+u32(position.x);
}
// One word per pixel past every candidate slot of the shared arena. Both
// coverage passes address the arena from zero with their own stride, and the
// wider of the two (the scene-primitive capacity) sizes the allocation, so this
// tail is unreachable from either — see svoRasterCoverageArenaBytes.
const DRY_SCENE_COVERAGE_CAPACITY:u32=${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel}u;
const DRY_SCENE_COVERAGE_INDEX_MASK:u32=${SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS - 1}u;
fn dryPrimaryCoverageSeedIndex(pixel:u32)->u32{
  return max(u32(uniforms.viewport.x),1u)*max(u32(uniforms.viewport.y),1u)*DRY_SCENE_COVERAGE_CAPACITY+pixel;
}
fn dryBrickCoverageExactHit(position:vec2f,ro:vec3f,rd:vec3f,tLimit:f32)->DryHit{
  let pixel=dryBrickCoveragePixel(position);var opaque=missHit();opaque.t=tLimit;
  if(pixel>=dryCoveragePixelLimit()){return opaque;}
  let count=min(atomicLoad(&svoBrickCoverageCounts[pixel]),${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u);
  let base=pixel*${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u;var visited=0u;var iteration=0u;
  // One ray/AABB intersection per candidate, not one per candidate per pass.
  //
  // The selection below is a visited-mask extraction and is O(n^2) in
  // comparisons by construction (n <= the arena's per-pixel capacity). What it
  // used to put inside that square was a full svoRayAabbWithInverse plus a
  // 32-byte svoBrickInstances fetch for every unvisited slot on every pass —
  // both loop-invariant, so n^2 intersections and n^2 record loads to answer a
  // question n of each already answers. Cache the interval once and let the
  // square touch registers only.
  //
  // Bit-identical to what it replaces: a slot that fails either bounds check
  // still caches DRY_MISS and can never beat an initial bestEntry of DRY_MISS,
  // the strict entry<bestEntry test keeps the earlier slot on a tie, and the
  // selected slot's instance is still re-read unguarded because it can only
  // have been selected when both checks passed.
  var slotEntry:array<f32,${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}>;
  var slotExit:array<f32,${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}>;
  for(var slot=0u;slot<${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u;slot+=1u){
    if(slot>=count){break;}
    var entry=DRY_MISS;var exit=DRY_MISS;
    let address=base+slot;
    if(address<arrayLength(&svoBrickCoverageCandidates)){
      let instanceIndex=svoBrickCoverageCandidates[address];
      if(instanceIndex<arrayLength(&svoBrickInstances)){
        let record=svoBrickInstances[instanceIndex];
        let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(record.proxyMinimum,record.proxyMaximum));
        entry=select(DRY_MISS,max(interval.y,0.0),interval.x!=0.0);exit=interval.z;
      }
    }
    slotEntry[slot]=entry;slotExit[slot]=exit;
  }
  while(iteration<count){
    var bestSlot=0xffffffffu;var bestEntry=DRY_MISS;var bestExit=DRY_MISS;var slot=0u;
    while(slot<count){
      if((visited&(1u<<slot))==0u){
        let entry=slotEntry[slot];
        if(entry<bestEntry){bestEntry=entry;bestExit=slotExit[slot];bestSlot=slot;}
      }
      slot+=1u;
    }
    if(bestSlot==0xffffffffu||bestEntry>=opaque.t){break;}visited|=1u<<bestSlot;
    let instanceIndex=svoBrickCoverageCandidates[base+bestSlot];let record=svoBrickInstances[instanceIndex];
    let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,0u,record.voxelOffset,0u,bestEntry,bestExit);
    let payload=traceLeafPayload(ro,rd,leaf);
    if(payload.t<DRY_MISS){if(payload.t<opaque.t){opaque=payload;}break;}
    iteration+=1u;
  }
  return opaque;
}
fn dryBrickCoverageResolve(position:vec2f)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  var opaque=missHit();var producer=SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND;
  ${screenSpaceBrickResolveWGSL}
  let pixel=dryBrickCoveragePixel(position);let exact=dryBrickCoverageExactHit(position,ro,rd,opaque.t);
  if(exact.t<opaque.t){opaque=exact;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  // The rigid impostor fold. Raster-primary's background pass draws nothing at
  // all now, which is the whole reason a scene with any body at all had to keep
  // the twelve-instance impostor pass switched on — and that pass, not the
  // bodies, is what forfeited stationary primary reuse. One analytic body loop inside
  // the resolve that already owns this pixel costs a bounding-sphere reject on
  // every ray that misses the set, and it publishes bodies through exactly the
  // same G-buffer encoder as every other producer here.
  if(svoRigidBoundsIntersect(ro,rd,opaque.t)){
    let rigid=nearestBody(ro,rd);
    if(rigid.t<opaque.t){opaque=rigid;producer=SVO_GBUFFER_PRODUCER_RIGID;}
  }
  // Publish this pixel's nearest opaque for the scene-primitive resolve. One
  // full-screen invocation per pixel writes it exactly once, so the store needs
  // no atomic and races with nothing. The brick overflow arm can still find a
  // nearer brick afterwards, which only makes the seed conservative: it is an
  // upper bound on the true nearest, so it can never cull a winning candidate.
  let seedIndex=dryPrimaryCoverageSeedIndex(pixel);
  if(seedIndex<arrayLength(&svoBrickCoverageCandidates)){svoBrickCoverageCandidates[seedIndex]=bitcast<u32>(opaque.t);}
  if(opaque.t<DRY_MISS){return dryRasterPrimarySurface(opaque,ro,rd,camera[1],producer);}
  return dryRasterPrimaryMiss();
}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.resolve}(input:VertexOut)->DryRasterPrimaryOut{
  return dryBrickCoverageResolve(input.position.xy);
}
${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
// The production threshold keeps the heavyweight exact marcher out of sky and
// sub-pixel tiles entirely. Entry-point reachability gives each tier its own
// Metal register budget even though the source module remains shared.
@fragment fn svoBrickLodResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let lodKey=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(lodKey==0u){discard;}let instanceIndex=lodKey-1u;if(instanceIndex>=arrayLength(&svoBrickInstances)){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let record=svoBrickInstances[instanceIndex];let bounds=mat2x3f(record.proxyMinimum,record.proxyMaximum);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,bounds);if(interval.x==0.0){discard;}
  let proxy=dryPrimaryLeafProxyHit(ro,rd,record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK,bounds,max(interval.y,0.0),interval.z,record.voxelOffset);
  if(!(proxy.t<DRY_MISS)){discard;}return dryRasterPrimarySurface(proxy,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
@fragment fn svoBrickExactResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let pixel=dryBrickCoveragePixel(position);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])==0u){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let exact=dryBrickCoverageExactHit(position,ro,rd,DRY_MISS);if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
` : ""}
// Overflow is correctness-only and isolated in its own proxy entry point. It
// repeats the historical direct fragment only on marked pixels, which makes
// capacity performance-only without importing canonical brick-boundary tie
// arithmetic into the raster arm.
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.overflowResolve}(input:SvoBrickRasterVertexOut)->DryRasterPrimaryOut{
  let pixel=dryBrickCoveragePixel(input.position.xy);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])<=${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(input.proxyMinimum,input.proxyMaximum));
  if(interval.x==0.0){discard;}
  let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,input.nodeIndex,0u,input.voxelOffset,0u,max(interval.y,0.0),interval.z);
  let payload=traceLeafPayload(ro,rd,leaf);if(!(payload.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
}
// ---------------------------------------------------------------------------
// The authored SDF set on the same coverage/resolve/overflow shape as bricks.
//
// The direct fragment above it marches the record's field once per covering
// proxy, and it writes frag_depth *and* discards, so a tiler can reject nothing:
// a covered pixel under a median of seven proxies marches all seven even after
// the nearest one has resolved it. WGSL has no conservative depth to promise
// monotonicity with (gpuweb#5342), so the substitute is the arena the brick
// raster already proves out: append candidates without evaluating any field,
// then walk one per-pixel list front to back exactly once.
// ---------------------------------------------------------------------------
fn dryScenePrimitiveCoverageKey(entry_m:f32,primitiveIndex:u32)->u32{
  return ((bitcast<u32>(max(entry_m,0.0))>>16u)<<16u)|(primitiveIndex&DRY_SCENE_COVERAGE_INDEX_MASK);
}
/** The key's floored entry distance: a lower bound, which is what makes the early-out sound. */
fn dryScenePrimitiveCoverageEntry(key:u32)->f32{return bitcast<f32>(key&0xffff0000u);}
// Coverage evaluates no field. It repeats only the oriented-box bracket the
// resolve will march inside, which is a quaternion rotate and a slab test, and
// that bracket is also the sort key. The three rejections here (retired index,
// suppressed owner, ray misses the local box) are exactly the ones the direct
// fragment discards on, so a candidate that never enters the arena is one the
// historical pass would have thrown away after paying for it.
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.coverage}(
  input:DryScenePrimitiveVertexOut,
 )${sceneCoverageReturnTypeWGSL}{
  if(input.primitiveIndex>=dry.metadata.x){${sceneCoverageMissWGSL}}
  let width=max(u32(uniforms.viewport.x),1u);let coordinate=vec2u(input.position.xy);
  let pixel=coordinate.y*width+coordinate.x;
  if(pixel>=dryCoveragePixelLimit()){${sceneCoverageMissWGSL}}
  dryRasterPrimaryReset();
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){${sceneCoverageMissWGSL}}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let span=dryScenePrimitiveMarchSpan(record,ro,rd);
  if(!(span.x<=span.y)){${sceneCoverageMissWGSL}}
  ${screenSpaceSceneCoverageWGSL}
  // Occlusion, one pass earlier than the resolve applies it.
  //
  // The brick/terrain/body resolve has already published this pixel's nearest
  // opaque, and after the voxel primary that surface is usually an authored
  // record's own exact hit rather than a miss. A proxy whose entry is at or
  // beyond it cannot hold a nearer surface — the identical front-to-back
  // argument the resolve's walk makes — so it is not a candidate at all.
  //
  // Rejecting it here rather than there is what keeps the arena from filling:
  // capacity is per pixel and unconditional, so a candidate the resolve would
  // have skipped in one comparison still consumed a slot, and slots are what the
  // overflow pass re-marches. This is image-exact — a rejected candidate is one
  // whose conservative lower bound is already behind a resolved surface.
  ${sceneCoverageSeedCullWGSL}
  let slot=atomicAdd(&svoBrickCoverageCounts[pixel],1u);
  ${screenSpaceTerminationPixels > 0 ? `if(dryCoveragePixelLimit()>=${1 << 20}u&&slot==0u){let queueIndex=atomicAdd(&dryTieredResolveQueue.pixelCount,1u);if(queueIndex<dryCoveragePixelLimit()){dryTieredResolveQueue.pixels[queueIndex]=pixel;}}` : ""}
  if(slot<DRY_SCENE_COVERAGE_CAPACITY){
    let address=pixel*DRY_SCENE_COVERAGE_CAPACITY+slot;
    if(address<arrayLength(&svoBrickCoverageCandidates)){
      svoBrickCoverageCandidates[address]=dryScenePrimitiveCoverageKey(span.x,input.primitiveIndex);
    }
    discard;
  }
  if(slot==DRY_SCENE_COVERAGE_CAPACITY){svoRasterCoverageOverflowSignal();}
  ${sceneCoverageOverflowWGSL}
}
// One expensive fragment per pixel. Candidates are extracted in increasing key
// order — unique, because back-face culling gives a primitive at most one
// fragment per pixel — so no visited set is needed and the walk is not bounded
// by a 32-bit mask the way the brick resolve's is.
fn dryScenePrimitiveCoverageResolve(position:vec2f,ro:vec3f,rd:vec3f)->DryHit{
  var best=missHit();
  let pixel=dryBrickCoveragePixel(position);
  if(pixel>=dryCoveragePixelLimit()){return best;}
  let appended=atomicLoad(&svoBrickCoverageCounts[pixel]);
  dryScenePrimitiveAuditCoverage(pixel,appended);
  let count=min(appended,DRY_SCENE_COVERAGE_CAPACITY);
  if(count==0u){return best;}
  // Seeded with the surface the brick/terrain/body resolve already published:
  // this is the depth authority for the pixel at this point in the frame, and
  // marching past it is the work the direct pass could never avoid.
  var limit=DRY_MISS;
  ${sceneCoverageSeedLimitWGSL}
  ${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
  // Overflow is the deep tier: it walks the authored set at this pixel instead
  // of retaining a full-MRT overflow render pass in every frame.
  if(appended>DRY_SCENE_COVERAGE_CAPACITY){
    for(var recordIndex=0u;recordIndex<dry.metadata.x;recordIndex+=1u){
      if(!dryScenePrimitiveBandMember(recordIndex)){continue;}
      let record=dryPrimitive(recordIndex);if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){continue;}
      let span=dryScenePrimitiveMarchSpan(record,ro,rd);if(!(span.x<=span.y)||!(span.x<limit)){continue;}
      let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
      let worldExtent=dryScenePrimitiveWorldExtent(localExtent,record.orientation);let centre=svoPrimitiveCenter_m(record);
      if(dryPrimaryBoundsSubPixel(mat2x3f(centre-worldExtent,centre+worldExtent))){continue;}
      let exact=primitiveHit(record,ro,rd,span.x,min(span.y,limit));if(exact.t<limit){limit=exact.t;best=exact;}
    }
    return best;
  }` : ""}
  let base=pixel*DRY_SCENE_COVERAGE_CAPACITY;
  // The live keys, once. The monotone extraction below is O(n^2) in comparisons
  // by construction (n <= DRY_SCENE_COVERAGE_CAPACITY), and it was re-loading
  // the same arena word from storage on every one of those iterations. The keys
  // are immutable for the life of this fragment — the coverage pass that wrote
  // them has already completed — so a single linear read answers every pass.
  var keys:array<u32,${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel}>;
  for(var slot=0u;slot<DRY_SCENE_COVERAGE_CAPACITY;slot+=1u){
    if(slot>=count){break;}
    keys[slot]=svoBrickCoverageCandidates[base+slot];
  }
  var previousKey=0u;var started=false;
  for(var iteration=0u;iteration<DRY_SCENE_COVERAGE_CAPACITY;iteration+=1u){
    if(iteration>=count){break;}
    var nextKey=0xffffffffu;var found=false;
    for(var slot=0u;slot<DRY_SCENE_COVERAGE_CAPACITY;slot+=1u){
      if(slot>=count){break;}
      let key=keys[slot];
      if((!started||key>previousKey)&&key<nextKey){nextKey=key;found=true;}
    }
    if(!found){break;}
    previousKey=nextKey;started=true;
    // Front-to-back termination. Every remaining candidate enters at or beyond
    // this one, so none of them can hold a nearer surface than the best hit.
    if(!(dryScenePrimitiveCoverageEntry(nextKey)<limit)){break;}
    let record=dryPrimitive(nextKey&DRY_SCENE_COVERAGE_INDEX_MASK);
    let span=dryScenePrimitiveMarchSpan(record,ro,rd);
    ${screenSpacePrimitiveResolveWGSL}
    // Marched only over what is still visible: the far end is clamped to the
    // best hit so far, so a sphere trace never walks behind a resolved surface.
    let exact=primitiveHit(record,ro,rd,span.x,min(span.y,limit));
    if(exact.t<limit){limit=exact.t;best=exact;}
  }
  return best;
}
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.resolve}(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let exact=dryScenePrimitiveCoverageResolve(input.position.xy,ro,rd);
  if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
${screenSpaceTerminationPixels > 0 ? /* wgsl */ `
@compute @workgroup_size(1) fn svoScenePrimitiveTieredComputeArgs(){
  let count=min(atomicLoad(&dryTieredResolveQueue.pixelCount),dryCoveragePixelLimit());
  atomicStore(&dryTieredResolveQueue.dispatchX,(count+63u)/64u);
  atomicStore(&dryTieredResolveQueue.dispatchY,1u);atomicStore(&dryTieredResolveQueue.dispatchZ,1u);
}
@compute @workgroup_size(64) fn svoScenePrimitiveTieredComputeResolve(
  @builtin(global_invocation_id) id:vec3u,
){
  let dimensions=vec2u(max(u32(uniforms.viewport.x),1u),max(u32(uniforms.viewport.y),1u));
  let count=min(atomicLoad(&dryTieredResolveQueue.pixelCount),dryCoveragePixelLimit());if(id.x>=count){return;}
  let pixel=dryTieredResolveQueue.pixels[id.x];let pixelCoordinate=vec2u(pixel%dimensions.x,pixel/dimensions.x);
  let coordinate=vec2i(pixelCoordinate);let position=vec2f(pixelCoordinate)+vec2f(.5);dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let exact=dryScenePrimitiveCoverageResolve(position,ro,rd);if(!(exact.t<DRY_MISS)){return;}
  let hardwareDepth=dryHardwareDepth(exact.t,rd,camera[1]);
  if(!(hardwareDepth>textureLoad(dryTierCurrentDepth,coordinate,0))){return;}
  let surface=dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
  textureStore(dryTierPackedSurfaceWrite,coordinate,surface.packedSurface);
  textureStore(dryTierIdentityMediaWrite,coordinate,surface.identityMedia);
  textureStore(dryTierGeometryWrite,coordinate,surface.geometry);
  textureStore(dryTierOpaqueIdentityWrite,coordinate,vec4u(surface.opaqueIdentity,0u,0u));
  textureStore(dryTierDepthWrite,coordinate,vec4f(hardwareDepth,0.0,0.0,0.0));
}
@fragment fn svoScenePrimitiveLodResolveFragment(input:VertexOut)->DryRasterPrimaryOut{
  let position=input.position.xy;let key=textureLoad(drySplitGlassKeyRead,vec2i(position),0).x;
  if(key==0u){discard;}let recordIndex=key-1u;if(recordIndex>=dry.metadata.x){discard;}
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(position,camera);
  let record=dryPrimitive(recordIndex);let span=dryScenePrimitiveMarchSpan(record,ro,rd);if(!(span.x<=span.y)){discard;}
  let proxy=dryPrimaryPrimitiveProxyHit(record,ro,rd,span);return dryRasterPrimarySurface(proxy,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
` : ""}
// Correctness-only, exactly as the brick arm: the historical direct fragment,
// verbatim, on the pixels whose conservative list overflowed. Capacity is
// therefore a performance parameter and never an image change.
@fragment fn ${SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.overflowResolve}(
  input:DryScenePrimitiveVertexOut,
)->DryRasterPrimaryOut{
  let pixel=dryBrickCoveragePixel(input.position.xy);
  if(pixel>=dryCoveragePixelLimit()||atomicLoad(&svoBrickCoverageCounts[pixel])<=DRY_SCENE_COVERAGE_CAPACITY){discard;}
  dryRasterPrimaryReset();
  if(input.primitiveIndex>=dry.metadata.x){discard;}
  let record=dryPrimitive(input.primitiveIndex);
  if(dryOpaqueOwnerSuppressed(svoPrimitiveOwnerId(record))){discard;}
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  ${scenePrimitiveMarchSpanWGSL}
  if(!(exact.t<DRY_MISS)){discard;}
  return dryRasterPrimarySurface(exact,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
}
// Brick fragments carry their own output struct so the hidden-surface-removal
// probe can drop frag_depth here without disturbing the background pass, whose
// single full-screen fragment gains nothing from HSR either way.
struct DryBrickRasterOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,
  @location(3) opaqueIdentity:vec2u,${noFragmentDepth ? "" : `
  @builtin(frag_depth) hardwareDepth:f32,`}
}
fn dryBrickRasterOut(surface:DryRasterPrimaryOut)->DryBrickRasterOut{
  return DryBrickRasterOut(surface.packedSurface,surface.identityMedia,surface.geometry,surface.opaqueIdentity${noFragmentDepth ? "" : ",surface.hardwareDepth"});
}
@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.fragment}(input:SvoBrickRasterVertexOut)->DryBrickRasterOut{
  dryRasterPrimaryReset();
  let camera=dryRasterPrimaryCamera();let ro=camera[0];let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let interval=svoRayAabbWithInverse(SvoRay(ro,0.0,rd,DRY_MISS),1.0/rd,mat2x3f(input.proxyMinimum,input.proxyMaximum));
  ${hsrProbe ? "" : "if(interval.x==0.0){discard;}"}
  // The payload DDA is the production leaf tracer, unmodified: the raster stage
  // only replaces the search that found this leaf.
  let leaf=SvoTraversalHit(SVO_STATUS_HIT,0u,input.nodeIndex,0u,input.voxelOffset,0u,max(interval.y,0.0),interval.z);
  let payload=traceLeafPayload(ro,rd,leaf);
  ${hsrProbe
    ? `var probe=dryRasterPrimaryMiss();
  if(payload.t<DRY_MISS){probe=dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);}
  return dryBrickRasterOut(probe);`
    : `if(!(payload.t<DRY_MISS)){discard;}
  return dryBrickRasterOut(dryRasterPrimarySurface(payload,ro,rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK));`}
}
` : "";
  const recordInvalidAoPageWGSL = reduced && split
    ? "atomicAdd(&dryPrepassBoundaryQueue.invalidAoPages,1u);"
    : "";
  const recordInvalidDirectPageWGSL = reduced && split
    ? "atomicAdd(&dryPrepassBoundaryQueue.invalidDirectPages,1u);"
    : "";
  const prepassEntryWGSL = reduced ? /* wgsl */ `struct DryPrepassGeometryOut{@location(0) geometry:vec4f,@location(1) identity:u32}
@fragment fn dryPrepassGeometryMain(input:VertexOut)->DryPrepassGeometryOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  var output=DryPrepassGeometryOut(vec4f(0.0),0xffffffffu);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)==0u||!dryNodeMipReady()){return output;}
  let opaque=traceOpaqueScene(ro,rd);
  if(!(opaque.t<DRY_MISS)){return output;}
  let geometricNormal=normalize(opaque.normal);
  output.geometry=vec4f(opaque.t,dryPrepassEncodeNormal(geometricNormal),f32(dryPrepassHitMetadata(opaque)));
  output.identity=dryPrepassPackIdentity(opaque);
  return output;
}
fn dryPrepassTraceVisibility(opaque:DryHit,ro:vec3f,rd:vec3f)->vec2u{
  dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  ${voxelLightCache ? "dryVoxelLightConsumerEligible=select(0u,1u,opaque.motionKind==DRY_GBUFFER_MOTION_STATIC);" : ""}
  let position=ro+rd*opaque.t;let geometricNormal=normalize(opaque.normal);
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  // AO cones exclude rigid blockers; those stay exact at full resolution.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)!=0u){
    let radius=dryContactVisibilityRadius();
    if(radius>0.0){
      let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
      let origin=position+geometricNormal*cellScale*.2;
      let coneSampleCount=max(dry.tuningCounts1.z,dry.tuningCounts1.y);
      var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){
        if(sampleIndex>=coneSampleCount){break;}
        let direction=dryContactVisibilityDirection(geometricNormal,opaque.featureId,sampleIndex&1u);
        let rotated=select(direction,normalize(direction+cross(geometricNormal,direction)*.7),sampleIndex>=2u);
        // AO keeps near-surface self-occlusion by design: zero normal disables
        // the shadow cones' receiver-plane coverage suppression.
        let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);
        if(cone.valid==0u){${recordInvalidAoPageWGSL}return DRY_PREPASS_INVALID_PACKED;}
        visibility+=cone.transmittance;
      }
      visibility0.x=clamp(visibility/f32(coneSampleCount),0.0,1.0);
    }
  }
  // Per-light cone shadow terms for every shaded slot; area lights average two fixed samples.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.exactShadow}u)!=0u){
    let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
    for(var lightIndex=0u;lightIndex<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u;lightIndex+=1u){
      if(lightIndex>=lightCount){break;}
      let light=dryLighting.lights[lightIndex];
      if(light.identity.w!=dryLighting.metadata.y){continue;}
      let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA||light.identity.x==SVO_LIGHT_SPOT;
      let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;
      let sampleCount=select(select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area),1u,globalIllumination);
      ${voxelLightCache ? "dryCurrentLightSlot=lightIndex;let cachedVoxel=dryVoxelLightVisibility(position,geometricNormal);if(cachedVoxel.y>0.0){let packedVisibility=mix(1.0,cachedVoxel.x,dry.tuningRays0.y);if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}else{visibility2.x=packedVisibility;}continue;}" : ""}
      var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){
        if(sampleIndex>=sampleCount){continue;}
        let sample=dryLightSample(light,sampleIndex,position);
        if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}
        let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
        if(dryDirectionalRayLeavesDomain(maximumDistance)){visibility+=1.0;continue;}
        let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
        // Mirror of the inline path's normal escape and finite-emitter
        // clearance: the reduced-rate texel must hold the same visibility the
        // full-rate edge band computes inline.
        let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
        let coneEscape_m=coneCell_m*dry.tuningRays1.z;
        let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,sample.towardLight));
        let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,sample.finiteDistance_m>0.0);
        let cone=dryConeVisibility(ray.origin_m+geometricNormal*coneEscape_m,sample.towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,sample.finiteDistance_m>0.0);
        if(cone.valid==0u){${recordInvalidDirectPageWGSL}return DRY_PREPASS_INVALID_PACKED;}
        visibility+=mix(1.0,cone.transmittance,dry.tuningRays0.y);
      }
      let packedVisibility=clamp(visibility/f32(sampleCount),0.0,1.0);
      if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}
      else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}
      else{visibility2.x=packedVisibility;}
    }
  }
  return dryPrepassPack(visibility0,visibility1,visibility2);
}
const DRY_SILHOUETTE_EXACT_VALID:u32=0u;
const DRY_SILHOUETTE_EXACT_EXHAUSTED:u32=1u;
const DRY_SILHOUETTE_EXACT_INVALID:u32=2u;
const DRY_SILHOUETTE_REASON_NONE:u32=0u;
const DRY_SILHOUETTE_REASON_PUBLICATION:u32=1u;
const DRY_SILHOUETTE_REASON_SCENE_LIMIT:u32=2u;
const DRY_SILHOUETTE_REASON_TRAVERSAL:u32=3u;
const DRY_SILHOUETTE_REASON_TRACE_CONTRACT:u32=4u;
struct DrySilhouetteVisibility{packed:vec2u,status:u32,reason:u32}
fn drySilhouetteTraceVisibilityExact(opaque:DryHit,ro:vec3f,rd:vec3f)->DrySilhouetteVisibility{
  dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  let position=ro+rd*opaque.t;let geometricNormal=normalize(opaque.normal);
  // This explicit sparse edge tier is authoritative, so it uses the existing
  // hard visibility caps rather than the lower ordinary-frame quality budget.
  // Work remains strictly bounded to the compacted silhouette queue.
  let budget=SvoVisibilityBudget(${SVO_VISIBILITY_LIMITS.nodeVisits}u,${SVO_VISIBILITY_LIMITS.leafVisits}u,${SVO_VISIBILITY_LIMITS.workItems}u,${SVO_VISIBILITY_LIMITS.intersections}u);
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)!=0u){
    let radius=dryContactVisibilityRadius();if(radius>0.0){let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},opaque.featureId!=SVO_FEATURE_SMOOTH);var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,opaque.featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);dryVisibilityStepInvalidReason=DRY_SILHOUETTE_REASON_NONE;let result=svoTraceVisibility(ray,budget,true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_EXHAUSTED){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_EXHAUSTED,DRY_SILHOUETTE_REASON_NONE);}if(result.status==SVO_VIS_STATUS_INVALID){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_INVALID,select(dryVisibilityStepInvalidReason,DRY_SILHOUETTE_REASON_TRACE_CONTRACT,dryVisibilityStepInvalidReason==DRY_SILHOUETTE_REASON_NONE));}visibility+=dot(result.transmittance,vec3f(1.0/3.0));}
      visibility0.x=clamp(visibility/f32(${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}),0.0,1.0);
    }
  }
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.exactShadow}u)!=0u){let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
    for(var lightIndex=0u;lightIndex<${SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights}u;lightIndex+=1u){if(lightIndex>=lightCount){break;}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA||light.identity.x==SVO_LIGHT_SPOT;let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;let sampleCount=select(select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area),1u,globalIllumination);var visibility=0.0;
      for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=sampleCount){continue;}let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(geometricNormal,sample.towardLight)<=0.0){continue;}let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);if(dryDirectionalRayLeavesDomain(maximumDistance)){visibility+=1.0;continue;}let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,sample.towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);dryVisibilityIgnoredBody=opaque.ownerId;dryVisibilityStepInvalidReason=DRY_SILHOUETTE_REASON_NONE;let result=svoTraceVisibility(ray,budget,true,0.001,max(ray.originBias_m,1e-6));dryVisibilityIgnoredBody=DRY_OWNER_NONE;if(result.status==SVO_VIS_STATUS_EXHAUSTED){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_EXHAUSTED,DRY_SILHOUETTE_REASON_NONE);}if(result.status==SVO_VIS_STATUS_INVALID){return DrySilhouetteVisibility(DRY_PREPASS_INVALID_PACKED,DRY_SILHOUETTE_EXACT_INVALID,select(dryVisibilityStepInvalidReason,DRY_SILHOUETTE_REASON_TRACE_CONTRACT,dryVisibilityStepInvalidReason==DRY_SILHOUETTE_REASON_NONE));}visibility+=dot(result.transmittance,vec3f(1.0/3.0));}
      let packedVisibility=clamp(visibility/f32(sampleCount),0.0,1.0);if(lightIndex<3u){visibility0[1u+lightIndex]=packedVisibility;}else if(lightIndex<7u){visibility1[lightIndex-3u]=packedVisibility;}else{visibility2.x=packedVisibility;}
    }
  }
  return DrySilhouetteVisibility(dryPrepassPack(visibility0,visibility1,visibility2),DRY_SILHOUETTE_EXACT_VALID,DRY_SILHOUETTE_REASON_NONE);
}
@fragment fn dryPrepassVisibilityMain(input:VertexOut)->@location(0) vec2u{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec2u(0xffffffffu);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  return dryPrepassTraceVisibility(opaque,ro,rd);
}
fn dryPrepassShadeNoGi(opaque:DryHit,ro:vec3f,rd:vec3f)->vec3f{
  let position=ro+rd*opaque.t;let surface=dryEvaluateSurfaceMaterial(opaque,position);
  if(surface.valid==0u){return vec3f(0.0);}
  let closure=unifiedPbrMaterial(surface.baseColor,surface.metallic,surface.roughness,vec3f(0.0),0.0,surface.specularF0,surface.specularWeight,vec3f(0.0),0.0);
  var direct=vec3f(0.0);var sampleBudget=0u;let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
  for(var lightIndex=0u;lightIndex<${SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u;lightIndex+=1u){
    if(lightIndex>=lightCount||sampleBudget>=dry.tuningCounts0.z){break;}let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}
    let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA||light.identity.x==SVO_LIGHT_SPOT;let sampleCount=select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area);
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){
      if(sampleIndex>=sampleCount||sampleBudget>=dry.tuningCounts0.z){break;}sampleBudget+=1u;let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(opaque.normal,sample.towardLight)<=0.0){continue;}
      let maximumDistance=select(directionalLightSceneExitDistance(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
      let rigidBlocked=anyBodyBlockerIgnoring(position,sample.towardLight,opaque.ownerId,maximumDistance);let raw=select(dryPrepassChannel(1u+lightIndex),0.0,rigidBlocked);let visibility=vec3f(mix(1.0,raw,dry.tuningRays0.y));
      let lighting=unifiedLightingInputWithGeometry(opaque.normal,opaque.normal,-rd,sample.towardLight,sample.radiance*visibility/f32(sampleCount));direct+=shadeUnifiedSurface(closure,lighting);
    }
  }
  let viewDirection=normalize(-rd);let reflected=reflect(rd,opaque.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let environmentBrdf=unifiedEnvironmentBrdf(max(dot(opaque.normal,viewDirection),0.0),surface.roughness,f0);let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);
  let diffuseVisibility=dryDiffuseMultiBounceVisibility(dryPrepassData0.x,diffuseColor);let diffuseEnvironment=diffuseColor*diffuseEnergy*svoEnvironmentDiffuseIrradiance(dryLighting.environment,opaque.normal)*diffuseVisibility/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*environmentBrdf;
  return max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*dry.giLighting.w,vec3f(0.0));
}
@fragment fn dryPrepassShadeMain(input:VertexOut)->@location(0) vec4f{
  let coordinate=vec2i(input.position.xy);let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){return vec4f(0.0);}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  // Until GLOBAL data is ready, rigid opaque radiance remains exact at full
  // rate, so avoid doing an unusable complete material evaluation here.
  if(opaque.motionKind!=DRY_GBUFFER_MOTION_STATIC){return vec4f(0.0);}
  let packed=textureLoad(dryPrepassVisibilityKeyTexture,coordinate,0);let packedValid=!all(packed.xy==DRY_PREPASS_INVALID_PACKED);dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);
  dryPrepassState=select(0u,1u,packedValid);dryPrepassRadianceState=0u;dryPrepassGiState=0u;if(!packedValid){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u;return vec4f(0.0,0.0,0.0,-1.0);}dryCurrentLightSlot=0xffffffffu;dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;
  // Full-res relight stores only the expensive GI closure and evaluates the
  // material/direct/environment terms at every receiver. Reconstruction modes
  // instead cache the complete reduced-rate radiance: seed the private GI
  // closure first so shadeDryOpaque consumes this exact sample without tracing
  // it a second time. Invalid samples carry a negative alpha and are rejected
  // by the full-rate reconstruction, which then falls through to exact relight.
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)==0u){
    var depth=opaque.t;
    // The no-GI shortcut does not enter shadeDrySurface, so glass still needs
    // one continuation trace here. Store it beside the cached radiance; the
    // full-rate reconstruction then consumes this alpha without tracing again.
    if(dryHitThinDielectric(opaque)){let behind=dryTraceBeyondThinWall(opaque,ro,rd);depth=select(0.0,behind.t,behind.t<DRY_MISS);}
    return vec4f(dryPrepassShadeNoGi(opaque,ro,rd),depth);
  }
  {let ignoredBodyOwner=select(DRY_OWNER_NONE,opaque.ownerId,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(ro+rd*opaque.t,opaque.normal,ignoredBodyOwner);
    if(dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u
      ||dry.tuningCounts2.w==${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u){return vec4f(gi.radiance,select(-1.0,gi.visibility,gi.valid!=0u));}
    if(gi.valid==0u){return vec4f(0.0,0.0,0.0,-1.0);}dryPrepassGi=vec4f(gi.radiance,gi.visibility);dryPrepassGiState=1u;
  }
  let radiance=shadeDrySurface(opaque,ro,rd);
  return vec4f(radiance,drySurfaceOcclusionDepth_m);
}
` : "";
  const splitEntryWGSL = split ? /* wgsl */ `struct DryVisibilityOut{
  @location(0) packedSurface:vec4u,
  @location(1) identityMedia:vec4u,
  @builtin(frag_depth) hardwareDepth:f32,
}
fn drySplitVisibilityOut(targetsIn:SvoGBufferTargets,hardwareDepth:f32)->DryVisibilityOut{
  return DryVisibilityOut(targetsIn.packedSurface,targetsIn.identityMedia,hardwareDepth);
}
struct DryPrimarySeamSample{geometry:vec4f,identity:vec2u,valid:u32}
fn dryPrimarySeamForeground(depth:f32,centreDepth:f32)->bool{
  if(!(depth>0.0&&depth<DRY_MISS)){return false;}
  if(!(centreDepth>0.0&&centreDepth<DRY_MISS)){return true;}
  return depth+max(.001,.003*depth)<centreDepth;
}
fn dryPrimarySeamSample(coordinate:vec2i)->DryPrimarySeamSample{
  let dimensions=vec2i(textureDimensions(drySplitGeometryRead));
  let centre=drySplitGeometryAt(coordinate);let centreDepth=centre.w;
  let pairs=array<vec4i,4>(vec4i(-1,0,1,0),vec4i(0,-1,0,1),vec4i(-1,-1,1,1),vec4i(-1,1,1,-1));
  var bestCoordinate=vec2i(0);var bestDepth=DRY_MISS;var valid=0u;
  for(var pairIndex=0u;pairIndex<4u;pairIndex+=1u){
    let pair=pairs[pairIndex];let firstCoordinate=coordinate+pair.xy;let secondCoordinate=coordinate+pair.zw;
    if(any(firstCoordinate<vec2i(0))||any(firstCoordinate>=dimensions)||any(secondCoordinate<vec2i(0))||any(secondCoordinate>=dimensions)){continue;}
    let first=drySplitGeometryAt(firstCoordinate);let second=drySplitGeometryAt(secondCoordinate);
    if(!dryPrimarySeamForeground(first.w,centreDepth)||!dryPrimarySeamForeground(second.w,centreDepth)){continue;}
    let firstIdentity=drySplitIdentityAt(firstCoordinate);let secondIdentity=drySplitIdentityAt(secondCoordinate);let differentSurface=(firstIdentity.x&0x8000ffffu)!=(secondIdentity.x&0x8000ffffu)||(firstIdentity.y&0xffffu)!=(secondIdentity.y&0xffffu);if(!differentSurface){continue;}
    // Extend the rear of the two bracketing surfaces. This closes the exposed
    // background without growing the nearer silhouette over its neighbour.
    let candidateCoordinate=select(secondCoordinate,firstCoordinate,first.w>=second.w);let candidateDepth=max(first.w,second.w);
    if(candidateDepth<bestDepth){bestDepth=candidateDepth;bestCoordinate=candidateCoordinate;valid=1u;}
  }
  if(valid==0u){return DryPrimarySeamSample(vec4f(0.0,1.0,0.0,DRY_MISS),vec2u(0u),0u);}
  return DryPrimarySeamSample(drySplitGeometryAt(bestCoordinate),drySplitIdentityAt(bestCoordinate).xy,1u);
}
fn dryPrimarySeamHit(sample:DryPrimarySeamSample)->DryHit{
  let packedOpaqueMaterial=sample.identity.x;${splitOpaqueMaterialDecodeWGSL}
  let metadata=sample.identity.y;
  return DryHit(sample.geometry.w,normalize(sample.geometry.xyz),opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u,metadata&DRY_OPAQUE_FACE_MASK,0u));
}
@fragment fn dryPrimarySeamMain(input:VertexOut)->DryVisibilityOut{
  let coordinate=vec2i(input.position.xy);let seam=dryPrimarySeamSample(coordinate);if(seam.valid==0u){discard;}
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  let opaque=dryPrimarySeamHit(seam);let generation=dryPublicationGeneration();let voxelGlass=dryHitThinDielectric(opaque);let media=dryMediumPair(rd,opaque.normal,select(DRY_MEDIUM_OPAQUE,DRY_MEDIUM_GLASS,voxelGlass));let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let producer=select(SVO_GBUFFER_PRODUCER_TRACED,SVO_GBUFFER_PRODUCER_GLASS,voxelGlass);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(producer);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}let targets=svoGBufferSurface(vec3f(0.0),opaque.t,dryGeometricNormal(opaque),opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
  return drySplitVisibilityOut(targets,dryHardwareDepth(opaque.t,rd,forward));
}
@fragment fn dryVisibilityMain(input:VertexOut)->DryVisibilityOut{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;${primaryWorkMap ? "dryPrimaryWorkReset();" : ""}
  ${primaryEntrySeedLoadWGSL}
  let opaque=${splitPrimaryTraceWGSL};${primaryWorkMap ? "dryPrimaryWorkPublish(vec2i(input.position.xy),opaque);" : ""}${splitVisibilityGlassDiscoveryWGSL}
  ${splitVisibilityGlassReturnWGSL}
  if(opaque.t<DRY_MISS){let voxelGlass=dryHitThinDielectric(opaque);let media=dryMediumPair(rd,opaque.normal,select(DRY_MEDIUM_OPAQUE,DRY_MEDIUM_GLASS,voxelGlass));let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let producer=select(SVO_GBUFFER_PRODUCER_TRACED,SVO_GBUFFER_PRODUCER_GLASS,voxelGlass);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(producer);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}let targets=svoGBufferSurface(vec3f(0.0),opaque.t,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);return drySplitVisibilityOut(targets,dryHardwareDepth(opaque.t,rd,forward));}
  return drySplitVisibilityOut(svoGBufferMiss(vec3f(0.0),0u,generation,DRY_GBUFFER_NO_INTERSECTION,svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED)),0.0);
}
${reduced ? `@fragment fn dryReconstructedLightingMain(input:VertexOut)->@location(0) vec4f{
  let coordinate=vec2i(input.position.xy);let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w<DRY_MISS)){discard;}
  let opaqueIdentity=drySplitIdentityAt(coordinate);let packedOpaqueMaterial=opaqueIdentity.x;${splitOpaqueMaterialDecodeWGSL}let metadata=opaqueIdentity.y;let opaque=DryHit(geometry.w,geometry.xyz,opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u,metadata&DRY_OPAQUE_FACE_MASK,0u));
  dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);dryPrepassState=0u;dryPrepassRadianceState=0u;dryPrepassGiState=0u;dryPrepassExactEdgeState=0u;dryCurrentLightSlot=0xffffffffu;
  if(dryNodeMipReady()){dryPrepassResolve(input.position.xy,opaque.t,opaque.normal,opaque);}if(dryPrepassRadianceState!=1u){discard;}
  ${rasterGlassDiscovery ? "let glassKey=textureLoad(drySplitGlassKeyRead,coordinate,0).x;" : "let glassKey=(packedOpaqueMaterial>>16u)&0x1ffu;"}if(glassKey>0u){discard;}
  // Radiance and its water-sort depth are a single cached result. In
  // particular, a glass sample's alpha is the opaque surface behind the pane.
  let ndc=input.uv*2.0-1.0;let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(dryPrepassRadiance.rgb,vec3f(0.0))*vignette,dryPrepassRadiance.a);
}
` : ""}@fragment fn dryLightingMain(input:VertexOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;
  let coordinate=vec2i(input.position.xy);var geometry=drySplitGeometryAt(coordinate);var opaqueIdentity=drySplitIdentityAt(coordinate);if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.silhouetteRefinement}u)!=0u){let seam=dryPrimarySeamSample(coordinate);if(seam.valid!=0u){geometry=seam.geometry;opaqueIdentity=vec4u(seam.identity,0u,0u);}}var opaque=missHit();
  let packedOpaqueMaterial=opaqueIdentity.x;${splitOpaqueMaterialDecodeWGSL}if(geometry.w<DRY_MISS){let metadata=opaqueIdentity.y;opaque=DryHit(geometry.w,geometry.xyz,opaqueMaterial,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u,metadata&DRY_OPAQUE_FACE_MASK,0u));}
  ${prepassResolveCallWGSL}var glass=dryGlassMiss();${splitGlassKeyLoadWGSL}${reduced ? `if(dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["wide-relight"]}u&&dry.tuningCounts2.w!=${SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"]}u&&dryPrepassRadianceState==1u&&glassKey==0u){${experiments.singlePassReconstruction !== false ? "let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(dryPrepassRadiance.rgb,vec3f(0.0))*vignette,dryPrepassRadiance.a);" : "discard;"}}` : ""}if(glassKey>0u){let recordIndex=glassKey-1u;if(recordIndex<dry.glass.y){let record=dryGlassPane(recordIndex);let candidate=svoThinGlassIntersect(record,ro,rd,0.0,opaque.t,1e-6,record.extentIorEpsilon.w);if(candidate.valid!=0u){glass=DryGlassHit(candidate,recordIndex);}}}var color=shadeDrySurface(opaque,ro,rd);var depth=drySurfaceOcclusionDepth_m;let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;if(glassVisible){let glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(color*vignette,vec3f(0.0)),select(0.0,depth,depth<DRY_MISS));
}
@fragment fn drySkyLightingMain(input:VertexOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassFailure=0u;dryThickGlassEnabled=0u;
  let coordinate=vec2i(input.position.xy);var opaque=missHit();
  // Skipping the G-buffer identity load is most of what makes this entry cheap,
  // but without raster glass discovery the glass key is packed into that very
  // plane, so there it has to be read after all.
  ${rasterGlassDiscovery ? "" : "let packedOpaqueMaterial=drySplitIdentityAt(coordinate).x;"}
  var glass=dryGlassMiss();${splitGlassKeyLoadWGSL}if(glassKey>0u){let recordIndex=glassKey-1u;if(recordIndex<dry.glass.y){let record=dryGlassPane(recordIndex);let candidate=svoThinGlassIntersect(record,ro,rd,0.0,opaque.t,1e-6,record.extentIorEpsilon.w);if(candidate.valid!=0u){glass=DryGlassHit(candidate,recordIndex);}}}var color=shadeDrySurface(opaque,ro,rd);var depth=drySurfaceOcclusionDepth_m;
  if(glass.hit.valid!=0u&&glass.hit.t_m<opaque.t){let glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);return vec4f(max(color*vignette,vec3f(0.0)),select(0.0,depth,depth<DRY_MISS));
}
` : "";
  const prepassVisibilityStoreWGSL = coneFanout
    ? "textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(0xffffffffu));"
    : "textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(dryPrepassTraceVisibility(opaque,ro,rd),0u,0u));";
  const prepassFanoutDeclarationWGSL = coneFanout
    ? "@group(1) @binding(8) var dryPrepassFanoutReceiverWrite:texture_storage_2d<rgba32float,write>;"
    : "";
  const prepassFanoutMissStoreWGSL = coneFanout
    ? "textureStore(dryPrepassFanoutReceiverWrite,coordinate,vec4f(0.0));"
    : "";
  const prepassFanoutHitStoreWGSL = coneFanout
    ? "textureStore(dryPrepassFanoutReceiverWrite,coordinate,vec4f(opaque.t,normalize(opaque.normal)));"
    : "";
  const prepassFromPrimaryEntryWGSL = reduced && split ? /* wgsl */ `struct DryPrepassBoundaryQueue{
  count:atomic<u32>,invalidAoPages:atomic<u32>,invalidDirectPages:atomic<u32>,failedRefinements:atomic<u32>,coordinates:array<u32>
}
@group(1) @binding(4) var dryPrepassVisibilityWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(5) var dryPrepassGeometryWrite:texture_storage_2d<rgba16float,write>;
@group(1) @binding(6) var dryPrepassIdentityWrite:texture_storage_2d<r32uint,write>;
@group(1) @binding(7) var<storage,read_write> dryPrepassBoundaryQueue:DryPrepassBoundaryQueue;
${prepassFanoutDeclarationWGSL}
@group(1) @binding(9) var drySilhouetteRefinementWrite:texture_storage_2d<rg32uint,write>;
@group(1) @binding(11) var<storage,read_write> drySilhouetteDispatch:array<atomic<u32>>;
@group(1) @binding(12) var drySilhouetteRefinementStateWrite:texture_storage_2d<r32uint,write>;
@group(1) @binding(13) var<storage,read_write> drySilhouetteFailureReasons:array<atomic<u32>>;
@compute @workgroup_size(1) fn dryPrepassResetMain(){
  atomicStore(&dryPrepassBoundaryQueue.count,0u);atomicStore(&dryPrepassBoundaryQueue.invalidAoPages,0u);atomicStore(&dryPrepassBoundaryQueue.invalidDirectPages,0u);atomicStore(&dryPrepassBoundaryQueue.failedRefinements,0u);
}
fn dryPrepassRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));let ndc=uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());return mat2x3f(ro,rd);}
fn dryPrepassStore(coordinate:vec2i,opaque:DryHit,ro:vec3f,rd:vec3f){if(!(opaque.t<DRY_MISS)){textureStore(dryPrepassVisibilityWrite,coordinate,vec4u(0xffffffffu));textureStore(dryPrepassGeometryWrite,coordinate,vec4f(0.0));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(0xffffffffu));${prepassFanoutMissStoreWGSL}return;}textureStore(dryPrepassGeometryWrite,coordinate,vec4f(opaque.t,dryPrepassEncodeNormal(opaque.normal),f32(dryPrepassHitMetadata(opaque))));textureStore(dryPrepassIdentityWrite,coordinate,vec4u(dryPrepassPackIdentity(opaque),0u,0u,0u));${prepassFanoutHitStoreWGSL}${prepassVisibilityStoreWGSL}}
@compute @workgroup_size(8,8) fn dryPrepassCoherentMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(dryPrepassGeometryWrite);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);let ray=dryPrepassRay(globalId.xy,dimensions);
  let fullDimensions=textureDimensions(drySplitGeometryRead);let maximumCoordinate=vec2i(fullDimensions)-vec2i(1);let sampleBase=clamp(vec2i(floor((vec2f(globalId.xy)+vec2f(.5))*vec2f(fullDimensions)/vec2f(dimensions)-vec2f(.5))),vec2i(0),maximumCoordinate);
  var primaryGeometry:array<vec4f,4>;var primaryIdentity:array<vec4u,4>;var allMiss=true;
  for(var sample=0u;sample<4u;sample+=1u){let sourceCoordinate=clamp(sampleBase+vec2i(i32(sample&1u),i32(sample>>1u)),vec2i(0),maximumCoordinate);primaryGeometry[sample]=drySplitGeometryAt(sourceCoordinate);primaryIdentity[sample]=drySplitIdentityAt(sourceCoordinate);allMiss=allMiss&&!(primaryGeometry[sample].w<DRY_MISS);}
  if(allMiss){dryPrepassStore(coordinate,missHit(),ray[0],ray[1]);return;}
  var referenceGeometry=primaryGeometry[3];var referenceIdentity=primaryIdentity[3];var homogeneous=referenceGeometry.w<DRY_MISS;
  for(var sample=0u;sample<3u;sample+=1u){let geometry=primaryGeometry[sample];let hit=geometry.w<DRY_MISS;let sameIdentity=(primaryIdentity[sample].x&0x8000ffffu)==(referenceIdentity.x&0x8000ffffu)&&primaryIdentity[sample].y==referenceIdentity.y;let depthClose=abs(geometry.w-referenceGeometry.w)<=max(.0001,.01*referenceGeometry.w);let normalClose=dot(normalize(geometry.xyz),normalize(referenceGeometry.xyz))>=.9999;homogeneous=homogeneous&&hit&&sameIdentity&&depthClose&&normalClose;}
  if(!homogeneous){${inlineBoundaryWGSL}}
  let metadata=referenceIdentity.y;let packedMaterial=referenceIdentity.x;let material=select(packedMaterial&0xffffu,0x80000000u|(packedMaterial&0xffffu),(packedMaterial&0x80000000u)!=0u);let opaque=DryHit(referenceGeometry.w,normalize(referenceGeometry.xyz),material,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));dryPrepassStore(coordinate,opaque,ray[0],ray[1]);
}
@compute @workgroup_size(64) fn dryPrepassBoundaryMain(@builtin(global_invocation_id) globalId:vec3u){
  let queueCount=atomicLoad(&dryPrepassBoundaryQueue.count);if(globalId.x>=queueCount){return;}let dimensions=textureDimensions(dryPrepassGeometryWrite);let packedCoordinate=dryPrepassBoundaryQueue.coordinates[globalId.x];let coordinate=vec2u(packedCoordinate%dimensions.x,packedCoordinate/dimensions.x);let ray=dryPrepassRay(coordinate,dimensions);dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassEnabled=0u;let opaque=traceOpaqueScene(ray[0],ray[1]);dryPrepassStore(vec2i(coordinate),opaque,ray[0],ray[1]);
}
@compute @workgroup_size(1) fn drySilhouetteResetMain(){
  atomicStore(&dryPrepassBoundaryQueue.count,0u);atomicStore(&dryPrepassBoundaryQueue.invalidAoPages,0u);atomicStore(&dryPrepassBoundaryQueue.invalidDirectPages,0u);atomicStore(&dryPrepassBoundaryQueue.failedRefinements,0u);for(var reason=0u;reason<${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.words}u;reason+=1u){atomicStore(&drySilhouetteFailureReasons[reason],0u);}atomicStore(&drySilhouetteDispatch[0],0u);atomicStore(&drySilhouetteDispatch[1],0u);atomicStore(&drySilhouetteDispatch[2],0u);
}
fn drySilhouetteAmbiguous(coordinate:vec2i,dimensions:vec2u)->bool{
  let geometry=drySplitGeometryAt(coordinate);if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return false;}let identity=drySplitIdentityAt(coordinate);let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var index=0u;index<4u;index+=1u){let neighbourCoordinate=coordinate+offsets[index];if(any(neighbourCoordinate<vec2i(0))||any(neighbourCoordinate>=vec2i(dimensions))){continue;}let neighbour=drySplitGeometryAt(neighbourCoordinate);if(!(neighbour.w>0.0&&neighbour.w<DRY_MISS)){return true;}let neighbourIdentity=drySplitIdentityAt(neighbourCoordinate);let sameSurface=(identity.x&0x8000ffffu)==(neighbourIdentity.x&0x8000ffffu)&&(identity.y&0xffffu)==(neighbourIdentity.y&0xffffu);let depthClose=abs(geometry.w-neighbour.w)<=max(.001,.02*geometry.w);if(!sameSurface||!depthClose){return true;}}
  return false;
}
@compute @workgroup_size(8,8) fn drySilhouetteClassifyMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(drySilhouetteRefinementWrite);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);textureStore(drySilhouetteRefinementStateWrite,coordinate,vec4u(DRY_SILHOUETTE_STATE_UNTOUCHED));if(!drySilhouetteAmbiguous(coordinate,dimensions)){return;}let queueIndex=atomicAdd(&dryPrepassBoundaryQueue.count,1u);if(queueIndex<dimensions.x*dimensions.y){dryPrepassBoundaryQueue.coordinates[queueIndex]=globalId.y*dimensions.x+globalId.x;}
}
@compute @workgroup_size(1) fn drySilhouetteFinalizeMain(){
  let dimensions=textureDimensions(drySilhouetteRefinementWrite);let count=min(atomicLoad(&dryPrepassBoundaryQueue.count),dimensions.x*dimensions.y);let groupCount=(count+${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize - 1}u)/${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}u;let x=min(groupCount,65535u);var y=0u;if(x>0u){y=(groupCount+x-1u)/x;}atomicStore(&drySilhouetteDispatch[0],x);atomicStore(&drySilhouetteDispatch[1],y);atomicStore(&drySilhouetteDispatch[2],1u);
}
@compute @workgroup_size(${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}) fn drySilhouetteRefineMain(@builtin(global_invocation_id) globalId:vec3u,@builtin(num_workgroups) groups:vec3u){
  let queueIndex=globalId.x+globalId.y*groups.x*${SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize}u;let queueCount=atomicLoad(&dryPrepassBoundaryQueue.count);if(queueIndex>=queueCount){return;}let dimensions=textureDimensions(drySilhouetteRefinementWrite);let packedCoordinate=dryPrepassBoundaryQueue.coordinates[queueIndex];let coordinate=vec2u(packedCoordinate%dimensions.x,packedCoordinate/dimensions.x);let geometry=drySplitGeometryAt(vec2i(coordinate));if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return;}let identity=drySplitIdentityAt(vec2i(coordinate));let metadata=identity.y;let packedMaterial=identity.x;let material=select(packedMaterial&0xffffu,0x80000000u|(packedMaterial&0xffffu),(packedMaterial&0x80000000u)!=0u);let opaque=DryHit(geometry.w,normalize(geometry.xyz),material,metadata&0xffffu,(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u));let ray=dryPrepassRay(coordinate,dimensions);let exact=drySilhouetteTraceVisibilityExact(opaque,ray[0],ray[1]);if(exact.status==DRY_SILHOUETTE_EXACT_EXHAUSTED){atomicAdd(&dryPrepassBoundaryQueue.invalidAoPages,1u);atomicAdd(&dryPrepassBoundaryQueue.failedRefinements,1u);atomicAdd(&drySilhouetteFailureReasons[${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.exhaustedWord}u],1u);}else if(exact.status==DRY_SILHOUETTE_EXACT_INVALID){atomicAdd(&dryPrepassBoundaryQueue.invalidDirectPages,1u);atomicAdd(&dryPrepassBoundaryQueue.failedRefinements,1u);atomicAdd(&drySilhouetteFailureReasons[min(exact.reason,${SVO_DRY_SILHOUETTE_FAILURE_REASON_CONTRACT.traceContractWord}u)],1u);}textureStore(drySilhouetteRefinementWrite,vec2i(coordinate),vec4u(exact.packed,0u,0u));textureStore(drySilhouetteRefinementStateWrite,vec2i(coordinate),vec4u(select(DRY_SILHOUETTE_STATE_FAILED,DRY_SILHOUETTE_STATE_VALID,exact.status==DRY_SILHOUETTE_EXACT_VALID)));
}
` : "";
  const worldGiCacheHelpersWGSL = reduced && split ? /* wgsl */ `
struct DryWorldGiCacheMetadata{state:atomic<u32>,signature:u32}
struct DryWorldGiCachePayload{radianceRg:u32,radianceBv:u32}
struct DryWorldGiCache{
  metadata:array<DryWorldGiCacheMetadata,${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount}>,
  payload:array<DryWorldGiCachePayload,${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount}>,
}
struct DryWorldGiCacheKey{readyState:u32,signature:u32,start:u32}
struct DryWorldGiCacheLookup{value:DryGlobalIllumination,hit:u32,claimSlot:u32,claimState:u32}
struct DryWorldGiBodyInfluence{bodyMask:u32,movingMask:u32,signature:u32}
struct DryWorldGiFrame{
  bodySignature:u32,
  movingBodyCount:u32,
  bodyCount:u32,
  reserved:u32,
  invalidGiPages:atomic<u32>,
  reserved1:u32,
  reserved2:u32,
  reserved3:u32,
  cameraPosition:vec4f,
  cameraForwardAspect:vec4f,
  cameraRight:vec4f,
  cameraUp:vec4f,
  bodySignatures:array<u32,12>,
}
@group(2) @binding(7) var<storage,read_write> dryWorldGiCache:DryWorldGiCache;
@group(2) @binding(8) var dryWorldGiOutput:texture_storage_2d<rgba16float,write>;
@group(2) @binding(9) var<storage,read_write> dryWorldGiFrame:DryWorldGiFrame;
fn dryWorldGiHash(valueIn:u32)->u32{
  var value=valueIn;value^=value>>16u;value*=0x7feb352du;value^=value>>15u;value*=0x846ca68bu;return value^(value>>16u);
}
fn dryWorldGiHashAdd(hash:u32,value:u32)->u32{return dryWorldGiHash(hash^(value+0x9e3779b9u+(hash<<6u)+(hash>>2u)));}
fn dryWorldGiMorton4(value:vec3u)->u32{
  var morton=0u;
  for(var bit=0u;bit<4u;bit+=1u){
    morton|=((value.x>>bit)&1u)<<(bit*3u);
    morton|=((value.y>>bit)&1u)<<(bit*3u+1u);
    morton|=((value.z>>bit)&1u)<<(bit*3u+2u);
  }
  return morton;
}
fn dryWorldGiSpatialStart(quantized:vec3i)->u32{
  let coordinate=vec3u(max(quantized,vec3i(0)));
  let tileHash=dryWorldGiHashAdd(dryWorldGiHashAdd(coordinate.x>>4u,coordinate.y>>4u),coordinate.z>>4u);
  // Morton-local low bits keep neighbouring shader lanes in the same cache
  // lines; six hashed high bits distribute repeating world tiles.
  return ((tileHash&63u)<<12u)|dryWorldGiMorton4(coordinate&vec3u(15u));
}
fn dryWorldGiKey(position:vec3f,normalIn:vec3f,bodyNamespace:u32)->DryWorldGiCacheKey{
  let cell=max(dry.mapping.cellSize,vec3f(1e-6));let quantized=vec3i(floor((position-dry.nodeMipOrigin.xyz)/(cell*.25)));
  // Prepass normals are decoded normalized; preserving that contract avoids a
  // normalize in every cache query.
  let normal=normalIn;let normalByte=vec3u(round(clamp(normal*.5+vec3f(.5),vec3f(0.0),vec3f(1.0))*255.0));
  let packedNormal=normalByte.x|(normalByte.y<<8u)|(normalByte.z<<16u);
  var first=dryWorldGiHashAdd(0x811c9dc5u,bitcast<u32>(quantized.x));
  first=dryWorldGiHashAdd(first,bitcast<u32>(quantized.y));first=dryWorldGiHashAdd(first,bitcast<u32>(quantized.z));
  first=dryWorldGiHashAdd(first,packedNormal);first=dryWorldGiHashAdd(first,dry.nodeMip.x);
  first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giCones.x));first=dryWorldGiHashAdd(first,u32(round(dry.giCones.y)));
  first=dryWorldGiHashAdd(first,dry.tuningCounts0.y);first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giLighting.x));
  first=dryWorldGiHashAdd(first,bitcast<u32>(dry.giLighting.y));first=dryWorldGiHashAdd(first,dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion}u);
  let second=dryWorldGiHashAdd(dryWorldGiHash(first^0x68bc21ebu),bodyNamespace);
  let ready=(first&0x3fffffffu)|0x80000000u;return DryWorldGiCacheKey(ready,second,dryWorldGiSpatialStart(quantized));
}
fn dryWorldGiFind(key:DryWorldGiCacheKey)->DryWorldGiCacheLookup{
  var claimSlot=0xffffffffu;var claimState=0u;
  for(var probe=0u;probe<${SVO_DRY_WORLD_GI_CACHE_CONTRACT.probeCount}u;probe+=1u){
    let slot=(key.start+probe*probe)&${SVO_DRY_WORLD_GI_CACHE_CONTRACT.entryCount - 1}u;let state=atomicLoad(&dryWorldGiCache.metadata[slot].state);
    if(state==key.readyState&&dryWorldGiCache.metadata[slot].signature==key.signature){
      let payload=dryWorldGiCache.payload[slot];
      let rg=unpack2x16float(payload.radianceRg);let bv=unpack2x16float(payload.radianceBv);
      let verifiedState=atomicLoad(&dryWorldGiCache.metadata[slot].state);
      if(verifiedState==key.readyState&&dryWorldGiCache.metadata[slot].signature==key.signature){
        return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(rg,bv.x),bv.y,1u),1u,claimSlot,claimState);
      }
    }
    if(state==0u){claimSlot=slot;claimState=0u;break;}
    // A ready entry is safe to replace after a compare-exchange claim. Never
    // select state 1, which denotes a writer currently publishing its payload.
    if(state!=1u){claimSlot=slot;claimState=state;}
  }
  return DryWorldGiCacheLookup(DryGlobalIllumination(vec3f(0.0),1.0,1u),0u,claimSlot,claimState);
}
fn dryWorldGiInsert(key:DryWorldGiCacheKey,slot:u32,claimState:u32,value:DryGlobalIllumination){
  if(slot==0xffffffffu){return;}let claimed=atomicCompareExchangeWeak(&dryWorldGiCache.metadata[slot].state,claimState,1u);
  if(!claimed.exchanged){return;}dryWorldGiCache.metadata[slot].signature=key.signature;
  dryWorldGiCache.payload[slot].radianceRg=pack2x16float(value.radiance.xy);
  dryWorldGiCache.payload[slot].radianceBv=pack2x16float(vec2f(value.radiance.z,value.visibility));
  atomicStore(&dryWorldGiCache.metadata[slot].state,key.readyState);
}
fn dryWorldGiBodyInfluence(position:vec3f,ignoredBodyOwner:u32)->DryWorldGiBodyInfluence{
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  var bodyMask=0u;var movingMask=0u;var signature=0x4f1bbcdcu;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}if(bodyIndex==ignoredBodyOwner){continue;}let body=bodies[bodyIndex];
    let influence=body.positionRadius.w+max(
      minimumVoxel*${SVO_DRY_WORLD_GI_CACHE_CONTRACT.dynamicInfluenceCells}.0,
      body.positionRadius.w*${SVO_DRY_WORLD_GI_CACHE_CONTRACT.dynamicInfluenceBodyRadii}.0);
    let delta=position-body.positionRadius.xyz;
    if(dot(delta,delta)<=influence*influence){
      let bodyBit=1u<<bodyIndex;bodyMask|=bodyBit;
      signature=dryWorldGiHashAdd(signature,dryWorldGiFrame.bodySignatures[bodyIndex]);
      let motion=rigidMotion[bodyIndex];
      if(motion.linearVelocityDisplacement.w>1e-7||motion.angularVelocityAngle.w>1e-7){movingMask|=bodyBit;}
    }
  }
  return DryWorldGiBodyInfluence(bodyMask,movingMask,signature);
}
` : "";
  const worldGiCacheEntryWGSL = reduced && split ? /* wgsl */ `
@compute @workgroup_size(1) fn dryWorldGiFrameMain(){
  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  var signature=dryWorldGiHashAdd(0x27d4eb2du,bodyCount);var movingBodyCount=0u;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}let body=bodies[bodyIndex];let motion=rigidMotion[bodyIndex];
    var bodySignature=dryWorldGiHashAdd(0x85ebca6bu,bodyIndex);
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.positionRadius.w));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.halfSizeShape.w));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.x));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.y));
    bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.z));bodySignature=dryWorldGiHashAdd(bodySignature,bitcast<u32>(body.orientation.w));
    dryWorldGiFrame.bodySignatures[bodyIndex]=bodySignature;signature=dryWorldGiHashAdd(signature,bodySignature);
    if(motion.linearVelocityDisplacement.w>1e-7||motion.angularVelocityAngle.w>1e-7){movingBodyCount+=1u;}
  }
  let origin=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-origin);
  let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  dryWorldGiFrame.bodySignature=signature;dryWorldGiFrame.movingBodyCount=movingBodyCount;dryWorldGiFrame.bodyCount=bodyCount;atomicStore(&dryWorldGiFrame.invalidGiPages,0u);
  dryWorldGiFrame.cameraPosition=vec4f(origin,0.0);
  dryWorldGiFrame.cameraForwardAspect=vec4f(forward,uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov());
  dryWorldGiFrame.cameraRight=vec4f(right,0.0);dryWorldGiFrame.cameraUp=vec4f(up,0.0);
}
fn dryWorldGiFrameRay(coordinate:vec2u,dimensions:vec2u)->mat2x3f{
  let uv=vec2f((f32(coordinate.x)+.5)/f32(dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(dimensions.y));
  let ndc=uv*2.0-1.0;let rd=normalize(dryWorldGiFrame.cameraForwardAspect.xyz
    +dryWorldGiFrame.cameraRight.xyz*ndc.x*dryWorldGiFrame.cameraForwardAspect.w
    +dryWorldGiFrame.cameraUp.xyz*ndc.y*cameraTanHalfFov());
  return mat2x3f(dryWorldGiFrame.cameraPosition.xyz,rd);
}
@compute @workgroup_size(8,8) fn dryWorldGiCacheMain(@builtin(global_invocation_id) globalId:vec3u){
  let dimensions=textureDimensions(dryWorldGiOutput);if(any(globalId.xy>=dimensions)){return;}let coordinate=vec2i(globalId.xy);
  let geometry=textureLoad(dryPrepassGeometryTexture,coordinate,0);
  if(geometry.x<=0.0){textureStore(dryWorldGiOutput,coordinate,vec4f(0.0,0.0,0.0,1.0));return;}
  let identity=textureLoad(dryPrepassIdentityTexture,coordinate,0).x;let metadata=u32(round(geometry.w));
  let opaque=DryHit(geometry.x,dryPrepassDecodeNormal(geometry.yz),identity&0xffffu,identity>>16u,metadata&15u,(metadata>>4u)&15u,(metadata>>8u)&3u,(metadata>>10u)&1u,0.0,vec3u(0u));
  let ray=dryWorldGiFrameRay(globalId.xy,dimensions);let position=ray[0]+ray[1]*opaque.t;
  let ignoredBodyOwner=select(DRY_OWNER_NONE,opaque.ownerId,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);
  let influence=dryWorldGiBodyInfluence(position,ignoredBodyOwner);
  // Only a moving body's own bounded neighbourhood is recomputed. A cached
  // static-body neighbourhood is keyed solely by the bodies local to it, so an
  // unrelated moving body cannot invalidate that cache line.
  if(influence.movingMask!=0u){
    dryWorldGiIgnoreRigidBodies=0u;dryWorldGiBodyMask=influence.bodyMask;dryPrepassGiState=0u;
    let dynamicValue=dryGlobalIllumination(position,opaque.normal,ignoredBodyOwner);
    if(dynamicValue.valid==0u){atomicAdd(&dryWorldGiFrame.invalidGiPages,1u);}
    textureStore(dryWorldGiOutput,coordinate,vec4f(dynamicValue.radiance,select(-1.0,dynamicValue.visibility,dynamicValue.valid!=0u)));return;
  }
  let bodyAware=influence.bodyMask!=0u;let bodyNamespace=select(0x4f1bbcdcu,influence.signature,bodyAware);
  let key=dryWorldGiKey(position,opaque.normal,bodyNamespace);let cached=dryWorldGiFind(key);
  if(cached.hit!=0u){textureStore(dryWorldGiOutput,coordinate,vec4f(cached.value.radiance,cached.value.visibility));return;}
  dryWorldGiIgnoreRigidBodies=select(1u,0u,bodyAware);dryWorldGiBodyMask=influence.bodyMask;dryPrepassGiState=0u;
  let value=dryGlobalIllumination(position,opaque.normal,select(DRY_OWNER_NONE,ignoredBodyOwner,bodyAware));
  if(value.valid==0u){atomicAdd(&dryWorldGiFrame.invalidGiPages,1u);}else{dryWorldGiInsert(key,cached.claimSlot,cached.claimState,value);}
  textureStore(dryWorldGiOutput,coordinate,vec4f(value.radiance,select(-1.0,value.visibility,value.valid!=0u)));
}
` : "";
  let shader = /* wgsl */ `
${svoMaterialWGSL}
${svoThickGlassWGSL}
${svoGBufferWGSL}
${svoPrimitiveMotionWGSL}
// The value noise the field-program tape and the cluster fields are built on.
// It came in with svoProceduralMaterialWGSL — the per-pixel material variation
// that was deleted with the rest of the world-position shading — and is retained
// on its own because the *geometry* still needs it.
${svoProceduralNoiseWGSL}
${svoLightWGSL}
${svoEnvironmentLightingWGSL}
${svoNodeMipSamplingWGSL}
${svoTetrahedralRadianceWGSL}
${svoTetrahedralRadianceConeCoreWGSL}
${svoFluidCoverageWGSL}
${planarBoundaryWGSL}
// highlight is (firstOwner, lastOwner, strength, falloff) for the object under
// the editor cursor, appended without moving any earlier uniform lane. A range
// rather than one id because a described object is
// several primitives — a lantern is three, a grown tree is thirty — and they are
// contiguous in owner order by construction. See lib/scenery-expand.ts.
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, reservedSceneMeta:vec4f, reservedSceneLanes:array<vec4f,16>, highlight:vec4f }
struct BodyGPU { positionRadius:vec4f, halfSizeShape:vec4f, orientation:vec4f, colorSelected:vec4f }
struct DryParams {
  mapping:SvoMapping,
  metadata:vec4u,
  lightDirection:vec4f,
  lightColor:vec4f,
  // x: reserved; y: environment/SolidWorld face-worklist count; zw: reserved.
  glass:vec4u,
  // x: record count; y: accepted generation; z: 64-byte record stride.
  planarBoundaries:vec4u,
  // x: dense slot count; y: table revision; z: 96-byte stride; w: bounded contact-visibility gate.
  materialPublication:vec4u,
  // x: stable address-plan generation; y: directory pages; z: levels; w: publication mode.
  nodeMip:vec4u,
  // xyz: opacity atlas texels; w: finest level that owns an opacity page.
  nodeMipAtlas:vec4u,
  // x: derived generation; y: canonical source generation; z: pages; w: descriptors.
  wideFanout:vec4u,
  // Twelve per-level directory row starts (count of pages with level < i) as three vec4u.
  nodeMipLevelStart:array<vec4u,3>,
  // xyz: world-space origin of the node-mip lattice.
  nodeMipOrigin:vec4f,
  // Dense evolving-fluid coverage volume; valid=0 skips every fluid sample site.
  fluidCoverage:SvoFluidCoverageFrame,
  // Runtime work caps: primary leaves, cone steps, lights, stable area samples.
  tuningCounts0:vec4u,
  // Moving area, stable/moving AO, exact visibility node visits.
  tuningCounts1:vec4u,
  // Exact visibility leaves, work items, intersections, radiance reconstruction mode.
  tuningCounts2:vec4u,
  // Shadow bias/strength and AO radius/strength.
  tuningRays0:vec4f,
  // AO aperture, shadow aperture, normal escape, emitter clearance.
  tuningRays1:vec4f,
  // Packed Z slabs in nodeMipPageTable. w=0 retains sorted-directory fallback.
  nodeMipDirect:vec4u,
  nodeMipDirectLevelZ:array<vec4u,3>,
  // x: tetrahedral-radiance generation; y: complete and generation-matched.
  tetrahedralRadiance:vec4u,
  nodeMipExtent:vec4f,
  // Bounce exposure, broad occlusion, diffuse environment, direct key.
  giLighting:vec4f,
  // Aperture, cone count, reserved, reserved.
  giCones:vec4f,
  // Centre and radius of one sphere enclosing every rigid body; radius < 0 when
  // the scene has none.
  rigidBounds:vec4f,
  // x: first BVH record; y: node count; z: root node; w: complete render revision.
  primitiveCandidates:vec4u,
  // u32 word offsets of control, publication, nodes, and leaves in the structural arena.
  structureOffsets:vec4u,
  // Optional derived traversal offsets; zero for canonical and compact modes.
  derivedTraversal:vec4u,
  // x: screen-space threshold in reference pixels; y: mode; z: fixed level;
  // w: filtered voxel-mesh detail threshold in reference pixels (zero: exact).
  lod:vec4f,
  // Banded lane bases inside the payload arena: occupancy, record mask, header, blob.
  payloadLanes:vec4u,
  // x: dense scene geometry; y: flat owner lane; z: voxel capacity; w: layout descriptor.
  payloadLanes1:vec4u,
  // x: primary entry-seed plane written this frame; zero means descend from the root.
  primaryEntry:vec4u,
  // enabled, normal strength, max coarsening level, hysteresis fraction.
  meshFilter:vec4f,
  // smooth normals enabled, minimum normal agreement, preserve close normals, reserved.
  meshFilterNormals:vec4f,
}
struct DryLightingArena {
  // x: light count; y: light revision; z: environment revision; w: environment ABI version.
  metadata:vec4u,
  lights:array<SvoLightRecord,${SVO_LIGHT_MAXIMUM_RECORDS}>,
  environment:SvoEnvironmentLightingRecord,
}
struct DryThickGlassArena {
  // x: count; y: revision; z: replaced thin-pane ID; w: binder ABI version.
  metadata:vec4u,
  records:array<SvoThickGlassRecord,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}>,
}
struct DryHit {
  t:f32,
  normal:vec3f,
  materialId:u32,
  ownerId:u32,
  featureId:u32,
  fieldSource:u32,
  motionKind:u32,
  motionValid:u32,
  selected:f32,
  // Three words the struct's alignment pays for anyway, so a hit may carry a
  // little more than the surface without costing a byte. \`x\` is the
  // screen-space proxy's depth band; \`y\` is the geometric face word a
  // G-buffer hit was published with (see DRY_OPAQUE_FACE_VALID) and is zero
  // for every hit that came from a trace rather than a read; \`z\` is unused.
  // A positional constructor passing \`vec3u(0u)\` states "none of the above",
  // which is what every producer but the voxel mesh means.
  aux:vec3u,
}

@group(0) @binding(0) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(1) var<uniform> bodies:array<BodyGPU,12>;
// The whole scene payload arena, not the owner lane's slice.
//
// Identity stopped being a lane when the banded leaf payload landed: it is an
// occupancy bit, a per-leaf header and a palette entry in four lanes of this one
// buffer, so the bases come from dry.payloadLanes and every read goes through
// sceneIdentityAt/sceneIdentityOf. The binding *type* is unchanged, which is
// what leaves the accepted planar catalogue as the only new storage binding.
@group(0) @binding(3) var<storage,read> scenePayload:array<u32>;
@group(0) @binding(4) var<storage,read> drySceneArena:array<u32>;
@group(0) @binding(6) var<storage,read> dryPlanarBoundaries:array<PlanarBoundaryPatch>;
@group(0) @binding(9) var<uniform> dry:DryParams;
@group(0) @binding(13) var<uniform> dryLighting:DryLightingArena;
@group(0) @binding(14) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>;
@group(0) @binding(15) var<uniform> thickGlass:DryThickGlassArena;
@group(0) @binding(16) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(17) var nodeMipSampler:sampler;
@group(0) @binding(18) var nodeMipDirectory:texture_2d<u32>;
@group(0) @binding(19) var fluidCoverageVolume:texture_3d<f32>;
@group(0) @binding(20) var nodeMipPageTable:texture_3d<u32>;
@group(0) @binding(21) var tetraRadianceLobe0:texture_3d<f32>;
@group(0) @binding(22) var tetraRadianceLobe1:texture_3d<f32>;
@group(0) @binding(23) var tetraRadianceLobe2:texture_3d<f32>;
@group(0) @binding(24) var tetraRadianceLobe3:texture_3d<f32>;
@group(0) @binding(25) var tetraRadianceBlackPages:texture_2d<u32>;
@group(0) @binding(26) var nodeMipPageValidity:texture_2d<u32>;
@group(0) @binding(27) var tetraRadiancePageValidity:texture_2d<u32>;
${sceneIdentityWGSL}

const DRY_SCENE_MATERIAL_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.materialOffsetBytes / 4}u;
const DRY_SCENE_PRIMITIVE_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.primitiveOffsetBytes / 4}u;
const DRY_SCENE_GLASS_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes / 4}u;
const DRY_SCENE_CLUSTER_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes / 4}u;
const DRY_SCENE_CLUSTER_WORD_LIMIT:u32=DRY_SCENE_CLUSTER_WORD_OFFSET+${SVO_DRY_SCENE_CLUSTER_ARENA_SIZE_BYTES / 4}u;
const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes / 4}u;
const DRY_SCENE_FIELD_PROGRAM_CAPACITY:u32=${SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY}u;
const DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS:u32=${SVO_FIELD_PROGRAM_BLOCK_WORDS}u;
// The sculpted-terrain region of the arena is still written by the producer and
// no longer addressed here: the shader that decoded it was the analytic ground.
// Its offsets stay in SVO_DRY_SCENE_ARENA_LAYOUT because every later region is
// placed after them, so reclaiming the bytes is an ABI change and not a deletion.
fn drySceneWords4(offset:u32)->vec4u{return vec4u(drySceneArena[offset],drySceneArena[offset+1u],drySceneArena[offset+2u],drySceneArena[offset+3u]);}
fn dryMaterial(index:u32)->SvoMaterialRecord{
  let base=DRY_SCENE_MATERIAL_WORD_OFFSET+index*${SVO_MATERIAL_RECORD_STRIDE_BYTES / 4}u;
  return SvoMaterialRecord(bitcast<vec4f>(drySceneWords4(base)),bitcast<vec4f>(drySceneWords4(base+4u)),
    bitcast<vec4f>(drySceneWords4(base+8u)),bitcast<vec4f>(drySceneWords4(base+12u)),
    bitcast<vec4f>(drySceneWords4(base+16u)),drySceneWords4(base+20u));
}
fn dryPrimitive(index:u32)->SvoPrimitiveRecord{
  let base=DRY_SCENE_PRIMITIVE_WORD_OFFSET+index*${SVO_PRIMITIVE_RECORD_STRIDE_BYTES / 4}u;
  return SvoPrimitiveRecord(drySceneWords4(base),drySceneWords4(base+4u),bitcast<vec4f>(drySceneWords4(base+8u)),drySceneWords4(base+12u));
}
// An aggregate's packing, resolved from the word-13 reference exactly as a
// heightfield's samples are. A record of any other kind carries the invalid
// sentinel there, so it reads the "not resolved" block and the ABI ignores it.
fn dryClusterPacking(record:SvoPrimitiveRecord)->SvoClusterPacking{
  if(svoPrimitiveKind(record)!=SVO_KIND_SMOOTH_UNION_CLUSTER){return svoInvalidClusterPacking();}
  let base=svoPrimitiveClusterReference(record);
  if(base<DRY_SCENE_CLUSTER_WORD_OFFSET||base+${SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS}u>DRY_SCENE_CLUSTER_WORD_LIMIT){return svoInvalidClusterPacking();}
  let head=drySceneWords4(base);
  let shape=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_LATTICE_LOBE_RADIUS_WORD}u));
  let placement=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_LOBE_SPAN_WORD}u));
  // The polyline is copied whatever the field is: a uniform read costs the same
  // for every cluster and branching on the field here would diverge the wave
  // for the one arm that needs it. Fields that do not use it never look.
  var points=array<vec4f,${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}>();
  for(var index=0u;index<${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}u;index+=1u){
    points[index]=bitcast<vec4f>(drySceneWords4(base+${CLUSTER_BLOCK_POINTS_WORD}u+index*4u));
  }
  return SvoClusterPacking(head[${CLUSTER_BLOCK_FIELD_WORD}],head[${CLUSTER_BLOCK_SEED_WORD}],head[${CLUSTER_BLOCK_COUNT_WORD}],
    bitcast<f32>(head[${CLUSTER_BLOCK_SMOOTH_RADIUS_WORD}]),shape.x,shape.y,shape.z,shape.w,
    placement.x,placement.y,placement.z,points);
}
fn drySceneWord(offset:u32)->u32{return drySceneArena[offset];}
${svoFieldProgramWGSL({
  functionName: "dryFieldProgramBlock",
  loadWord: "drySceneWord",
  baseWordExpression: "DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET",
  capacityExpression: "DRY_SCENE_FIELD_PROGRAM_CAPACITY",
})}
// The shared ABI's host hook. The generated evaluator above addresses tapes by
// slot; a record addresses its own by word offset, exactly as a heightfield and
// an aggregate do. Translating here is what keeps the reference *checkable*: an
// offset that is not a block start is a stale or corrupt word rather than a tape
// read from the middle of its neighbour, and it answers as unresolved — which
// the ABI reports as an invalid record instead of drawing the conservative box.
fn svoFieldProgramReferenceSample(reference:u32,localPoint:vec3f)->SvoFieldValue{
  if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET){return SvoFieldValue(1e20,1.0);}
  let offset=reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET;
  if(offset%DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS!=0u){return SvoFieldValue(1e20,1.0);}
  return dryFieldProgramBlock(offset/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,localPoint);
}
fn dryGlassPane(index:u32)->SvoThinGlassRecord{
  let base=DRY_SCENE_GLASS_WORD_OFFSET+index*${SVO_THIN_GLASS_RECORD_WORDS}u;
  return SvoThinGlassRecord(bitcast<vec4f>(drySceneWords4(base)),bitcast<vec4f>(drySceneWords4(base+4u)),
    bitcast<vec4f>(drySceneWords4(base+8u)),bitcast<vec4f>(drySceneWords4(base+12u)),drySceneWords4(base+16u));
}
fn dryPublicationWord(index:u32)->u32{return svoStructureWord(dry.structureOffsets.y+index);}

// Page failures remain typed inside the invocation so dependent cone/GI work
// can fail closed. They are diagnostics, not scene colour output.
var<private> dryDerivedPageFailure:u32=0u;

${canonicalTraversalWGSL}${screenSpaceTraversalWGSL}${lodDescentWGSL}${lodUniformWGSL}${screenSpacePrimaryProxyWGSL}${tieredComputeResolveDeclarationsWGSL}
${wideTraversalWGSL}${compactTraversalWGSL}${brickOccupancyHelpersWGSL}
${liveLeafLifecycleWGSL}
${createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, directPageTable: true })}
var<private> dryGiPageCache:DryNodeMipPageCache;
/** The ancestor page a sub-floor radiance sample redirects to; coarser, so it changes far less often. */
var<private> dryGiRadiancePageCache:DryNodeMipPageCache;
fn dryTetraRadianceReady()->bool{return dry.tetrahedralRadiance.y!=0u&&dry.tetrahedralRadiance.x==dry.nodeMip.x&&dryNodeMipReady();}
fn dryTetraRadiancePageValid(pageIndex:u32)->bool{
  let dimensions=textureDimensions(tetraRadiancePageValidity);return svoDerivedPageValidityResident(dimensions,pageIndex)&&textureLoad(tetraRadiancePageValidity,svoDerivedPageValidityTexel(dimensions,pageIndex),0).x!=0u;
}
fn dryNodeMipSceneExitDistance(position:vec3f,direction:vec3f)->f32{
  let minimum=dry.nodeMipOrigin.xyz;let maximum=minimum+dry.nodeMipExtent.xyz;var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){if(abs(direction[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/direction[axis];let second=(maximum[axis]-position[axis])/direction[axis];enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}}
  return max(exit,0.0);
}
/**
 * The radiance atlas's own page, which is not the opacity page.
 *
 * dry.tetrahedralRadiance.z is the finest level that owns a radiance page and
 * .w is the slot its atlas begins at. A cone sampling finer than the floor
 * reads the ancestor at the floor instead — legitimate, not a fallback:
 * planSvoNodeMipPyramid inserts an ancestor at every level above every occupied
 * base page, so residency is ancestor-closed and the ancestor is always there.
 * The lookup is recomputed from the query position at the floor level rather
 * than shifted from the fine page, so the fractional texel is exact instead of
 * quantised by the fine page's rounding.
 */
fn dryTetraRadianceLevelFloor()->u32{return min(dry.tetrahedralRadiance.z,dry.nodeMip.z-1u);}
/** 2: page unusable, 1: certified black, 0: sample it. Resolved once per page, never per tap. */
fn dryTetraRadianceState(pageIndex:u32)->u32{
  if(!dryTetraRadiancePageValid(pageIndex)){return 2u;}
  if(pageIndex<dry.tetrahedralRadiance.w){return 1u;}
  return textureLoad(tetraRadianceBlackPages,vec2u(pageIndex,0u),0).x;
}
fn dryTetraRadiancePageOrigin(pageIndex:u32)->vec3u{
  let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);
  let atlasPages=max(textureDimensions(tetraRadianceLobe0)/vec3u(physical),vec3u(1u));
  let slot=pageIndex-dry.tetrahedralRadiance.w;
  return vec3u(slot%atlasPages.x,(slot/atlasPages.x)%atlasPages.y,slot/(atlasPages.x*atlasPages.y))*physical;
}
fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
  if(!dryNodeMipReady()){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
  let level=min(max(u32(max(floor(query.lod),0.0)),dryNodeMipOpacityLevelFloor()),dry.nodeMip.z-1u);let levelScale=exp2(f32(level));
  let virtualVoxel=(query.position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*levelScale);let pageFloor=floor(virtualVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
  if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let pageCoordinate=vec3u(pageFloor);
  if(dryGiPageCache.generation!=dry.nodeMip.x||dryGiPageCache.level!=level||any(dryGiPageCache.coordinate!=pageCoordinate)){
    dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,0u);let pageIndex=dryNodeMipFind(level,pageCoordinate);
    if(pageIndex!=0xffffffffu){
      if(!dryNodeMipPageValid(pageIndex)){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
      var black=2u;if(dryTetraRadiancePageValid(pageIndex)){black=textureLoad(tetraRadianceBlackPages,vec2u(pageIndex,0u),0).x;}
      if(dry.nodeMipDirect.w!=0u){let physical=u32(SVO_NODE_MIP_PHYSICAL_SIZE);let atlasPages=max(dry.nodeMipAtlas.xyz/vec3u(physical),vec3u(1u));let atlasPage=vec3u(pageIndex%atlasPages.x,(pageIndex/atlasPages.x)%atlasPages.y,pageIndex/(atlasPages.x*atlasPages.y));dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,atlasPage*physical,dry.nodeMip.x,1u,pageIndex,black);}
      else{let entry=svoNodeMipDirectoryEntry(nodeMipDirectory,pageIndex);dryGiPageCache=DryNodeMipPageCache(pageCoordinate,level,entry.pageOrigin,entry.generation,1u,pageIndex,black);}
    }
  }
  if(dryGiPageCache.resident==0u){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  if(!dryNodeMipPageValid(dryGiPageCache.pageIndex)){return SvoTetraRadianceConeSourceSample(0.0,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),0u,0u);}
  let local=virtualVoxel-vec3f(pageCoordinate)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);
  let opacity=svoNodeMipSamplePage(nodeMipAtlas,nodeMipSampler,dryGiPageCache.pageOrigin,local);
  if(!dryTetraRadianceReady()){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,0u);}
  let radianceLevel=max(level,dryTetraRadianceLevelFloor());
  var radianceIndex=dryGiPageCache.pageIndex;var radianceLocal=local;var radianceState=dryGiPageCache.blackRadiance;
  if(radianceLevel!=level){
    let radianceScale=exp2(f32(radianceLevel));
    let radianceVoxel=(query.position_m-dry.nodeMipOrigin.xyz)/(dry.mapping.cellSize*radianceScale);
    let radiancePageFloor=floor(radianceVoxel/f32(SVO_NODE_MIP_INTERIOR_SIZE));
    if(any(radiancePageFloor<vec3f(0.0))){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
    let radiancePage=vec3u(radiancePageFloor);
    if(dryGiRadiancePageCache.generation!=dry.nodeMip.x||dryGiRadiancePageCache.level!=radianceLevel||any(dryGiRadiancePageCache.coordinate!=radiancePage)){
      dryGiRadiancePageCache=DryNodeMipPageCache(radiancePage,radianceLevel,vec3u(0u),dry.nodeMip.x,0u,0xffffffffu,1u);
      let ancestor=dryNodeMipFind(radianceLevel,radiancePage);
      if(ancestor!=0xffffffffu){dryGiRadiancePageCache=DryNodeMipPageCache(radiancePage,radianceLevel,dryTetraRadiancePageOrigin(ancestor),dry.nodeMip.x,1u,ancestor,dryTetraRadianceState(ancestor));}
    }
    if(dryGiRadiancePageCache.resident==0u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
    radianceIndex=dryGiRadiancePageCache.pageIndex;radianceState=dryGiRadiancePageCache.blackRadiance;
    radianceLocal=radianceVoxel-vec3f(radiancePage)*f32(SVO_NODE_MIP_INTERIOR_SIZE)-vec3f(.5);
  }
  if(radianceState==2u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,0u);}
  if(radianceState==1u){return SvoTetraRadianceConeSourceSample(opacity.solidMean,SvoTetraRadiance(vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0)),1u,1u);}
  let radianceOrigin=select(dryTetraRadiancePageOrigin(radianceIndex),dryGiRadiancePageCache.pageOrigin,radianceLevel!=level);
  let uv=svoNodeMipAtlasUv(radianceOrigin,radianceLocal,textureDimensions(tetraRadianceLobe0));
  return SvoTetraRadianceConeSourceSample(opacity.solidMean,svoTetraSample(tetraRadianceLobe0,tetraRadianceLobe1,tetraRadianceLobe2,tetraRadianceLobe3,nodeMipSampler,uv),1u,1u);
}
struct DryGlobalIllumination{radiance:vec3f,visibility:f32,valid:u32}
${worldGiCacheHelpersWGSL}
fn dryGlobalIllumination(position:vec3f,normal:vec3f,ignoredBodyOwner:u32)->DryGlobalIllumination{${experiments.globalIlluminationAbsent
    // The GI-absent variant: the stub is the exact value the uniform flag-off
    // path returns, so the image is unchanged and the whole gather below is
    // dead code the compiler removes from the deferred kernel.
    ? /* wgsl */ `
  return DryGlobalIllumination(vec3f(0.0),1.0,1u);`
    : /* wgsl */ `
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)==0u){return DryGlobalIllumination(vec3f(0.0),1.0,1u);}
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
  if(!dryTetraRadianceReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
  ${prepassGiShortcutWGSL}
  let minimumVoxel=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(normal)*minimumVoxel*max(dry.tuningRays1.z,1.0);var indirect=vec3f(0.0);var visibility=0.0;
  let coneCount=clamp(u32(round(dry.giCones.y)),3u,4u);let perConeBudget=max(1u,min(64u,dry.tuningCounts0.y)/coneCount);
  for(var coneIndex=0u;coneIndex<4u;coneIndex+=1u){
    if(coneIndex>=coneCount){break;}let direction=svoTetraRadianceHemisphereDirection(normal,coneIndex,coneCount,0.0);
    dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    dryGiRadiancePageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);
    let sceneExit=dryNodeMipSceneExitDistance(origin,direction);
    // The same reading the direct term needed: a zero-length interval means this
    // cone never enters the pyramid, so there is nothing along it to occlude
    // with and nothing to gather. Full visibility, no bounce — not the zero
    // transmittance an empty trace would otherwise report, which lands as
    // mix(1, 0, occlusionStrength) and dims every surface outside the domain to
    // a fifth of its ambient. On an infinite ground plane that draws the
    // container's footprint onto the floor as a bright rectangle.
    if(sceneExit<=0.0){visibility+=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);continue;}
    var rigidHit=missHit();
    if(dryWorldGiIgnoreRigidBodies==0u){rigidHit=nearestBodyMaskIgnoring(origin,direction,ignoredBodyOwner,dryWorldGiBodyMask);}
    let rigidBlocked=rigidHit.t<sceneExit;
    let result=svoTetraRadianceConeTrace(SvoTetraRadianceConeConfig(origin,direction,dry.giCones.x,minimumVoxel,min(sceneExit,rigidHit.t),perConeBudget,.995,.0039215686,1u));
    let weight=svoTetraRadianceHemisphereWeight(coneIndex,coneCount);
    if(result.valid==0u||result.missingRadianceSamples!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
    let finiteRadiance=all(result.radiance==result.radiance)&&all(abs(result.radiance)<vec3f(65504.0));
    let finiteVisibility=result.transmittance==result.transmittance&&abs(result.transmittance)<65504.0;
    if(!finiteRadiance||!finiteVisibility){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.globalIlluminationPage}u;return DryGlobalIllumination(vec3f(0.0),1.0,0u);}
    indirect+=result.radiance*weight;
    let visibleThroughStatic=result.transmittance;
    visibility+=select(visibleThroughStatic,0.0,rigidBlocked)*weight;
  }
  let occlusionStrength=select(0.0,dry.giLighting.y,(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion}u)!=0u);
  return DryGlobalIllumination(max(indirect,vec3f(0.0))*dry.giLighting.x,mix(1.0,clamp(visibility,0.0,1.0),occlusionStrength),1u);`}
}
fn dryDiffuseMultiBounceVisibility(visibilityIn:f32,albedoIn:vec3f)->vec3f{
  let visibility=clamp(visibilityIn,0.0,1.0);let albedo=clamp(albedoIn,vec3f(0.0),vec3f(1.0));
  let a=2.0404*albedo-vec3f(0.3324);let b=-4.7951*albedo+vec3f(0.6417);let c=2.7552*albedo+vec3f(0.6903);
  return clamp(max(vec3f(visibility),((visibility*a+b)*visibility+c)*visibility),vec3f(0.0),vec3f(1.0));
}
${prepassDeclarationsWGSL}${splitDeclarationsWGSL}${voxelLightCacheWGSL}${worldGiCacheEntryWGSL}fn dryDiagnosticControl()->u32{return u32(round(max(uniforms.options.x,0.0)));}
fn dryDiagnosticMaximumNodeVisits()->u32{return clamp(dryDiagnosticControl()&511u,1u,256u);}
fn dryDiagnosticMaximumDepth()->u32{return clamp(dryDiagnosticControl()>>9u,1u,21u);}
fn dryTraverse(ray:SvoRay,mapping:SvoMapping)->SvoTraversalHit{return svoTraverseWithDepthLimit(ray,mapping,dryDiagnosticMaximumDepth());}
${traversalCursorWGSL}
${primaryTraversalCursorWGSL}
${leafAccessWGSL}
${svoPrimitiveWGSL}
${unifiedLightingShaderLibrary}
${svoThinGlassWGSL}
${svoVisibilityPreludeWGSL}

const DRY_MISS:f32 = 3.402823e38;
const REQUIRED_FIELDS:u32 = ${SVO_DRY_SCENE_REQUIRED_VALID_FIELDS}u; // topology | voxel identity
const DRY_OWNER_NONE:u32=0xffffu;
const DRY_MEDIUM_GLASS:u32=2u;const DRY_MEDIUM_OPAQUE:u32=3u;
const DRY_GBUFFER_FIELD_ANALYTIC:u32=4u;
// SVO_GBUFFER_FIELD_SOURCES.structuralDiscrete — a cell face, not a solved
// surface. The voxels-only primary reports this for every opaque hit it makes.
const DRY_GBUFFER_FIELD_VOXEL:u32=1u;
const DRY_GBUFFER_MOTION_STATIC:u32=0u;const DRY_GBUFFER_MOTION_RIGID:u32=1u;
// A surface's *geometric* face, when it is not the normal the pixel shades with.
//
// The G-buffer contract has carried two oct8 normals since it was written, and
// every producer has written the same value into both. The voxel mesh's
// filtered detail is the first that cannot: a quad is rasterised as one flat
// face and shaded with a normal baked into the voxels behind it, and a shadow
// ray biased along the *baked* normal on a silhouette leaves through the cube
// the quad belongs to and reports its own solid as an occluder.
//
// The deferred lighting reads the split G-buffer through two planes — the f32
// geometry plane and this rg32uint identity plane — and the oct8 packedSurface
// plane is not bound to it at all, so the face travels here instead. The
// identity plane's metadata word ends at bit 26 (motion validity), leaving bits
// 27..30 free; a six-axis face is exactly three of them plus a present bit, so
// naming it costs no attachment, no binding and no bandwidth. The oct8
// geometric slot of packedSurface is written correctly all the same, for the
// readers of the published contract.
//
// Every other producer leaves these bits clear, and a reader that finds them
// clear falls back to the single normal it already had — the same float, not a
// requantised one, so nothing else in the frame moves by a bit.
const DRY_OPAQUE_FACE_VALID:u32=134217728u;const DRY_OPAQUE_FACE_SHIFT:u32=28u;const DRY_OPAQUE_FACE_MASK:u32=2013265920u;
// Zero unless the face differs from the shading normal *and* is exactly axis
// aligned. Anything else keeps the bits clear and reads back as the shading
// normal, which is precisely today's behaviour.
fn dryOpaqueFaceWord(geometricNormal:vec3f,shadingNormal:vec3f)->u32{
  if(all(geometricNormal==shadingNormal)){return 0u;}
  let magnitude=abs(geometricNormal);
  var axis=0u;if(magnitude.y>magnitude.x){axis=1u;}if(magnitude.z>magnitude[axis]){axis=2u;}
  if(magnitude[axis]!=1.0||magnitude.x+magnitude.y+magnitude.z!=1.0){return 0u;}
  return DRY_OPAQUE_FACE_VALID|((axis*2u+select(0u,1u,geometricNormal[axis]>0.0))<<DRY_OPAQUE_FACE_SHIFT);
}
fn dryOpaqueFaceNormal(word:u32)->vec3f{
  let code=(word>>DRY_OPAQUE_FACE_SHIFT)&7u;var normal=vec3f(0.0);
  normal[code/2u]=select(-1.0,1.0,(code&1u)!=0u);return normal;
}
// The normal a visibility ray leaves along: the published face where there is
// one, and otherwise the surface normal itself, unchanged.
fn dryGeometricNormal(hit:DryHit)->vec3f{
  if((hit.aux.y&DRY_OPAQUE_FACE_VALID)==0u){return hit.normal;}
  return dryOpaqueFaceNormal(hit.aux.y);
}
const DRY_GBUFFER_HARD_FEATURE:u32=256u;const DRY_GBUFFER_NO_INTERSECTION:u32=1u;
const DRY_GBUFFER_WORK_EXHAUSTED:u32=2u;const DRY_GBUFFER_INVALID_FIELD:u32=3u;
const DRY_REVERSED_Z_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
// The rigid body a visibility ray started on, so it cannot shadow itself.
//
// It was \`dryVisibilityIgnoredOwner\` and it carried two namespaces at once: a
// *record* owner id, for the analytic walk that has since been deleted, and a
// *body index* in 0..11, for the rigid loop below. A voxel surface has no owner
// any more, so the only thing that ever sets it is a rigid hit, and the two
// namespaces overlap numerically — body 3 and record 3 are different objects
// that compared equal. Named for the one meaning it has left.
var<private> dryVisibilityIgnoredBody:u32;var<private> dryVisibilityStepInvalidReason:u32;var<private> dryThickGlassEnabled:u32;var<private> dryThickGlassFailure:u32;
// The material traversal owns both radiance and the depth used to sort water.
// Keeping the latter invocation-local lets a thin-dielectric ray publish the
// opaque hit it already found instead of launching the same traversal twice.
var<private> drySurfaceOcclusionDepth_m:f32;
var<private> dryWorldGiIgnoreRigidBodies:u32;
var<private> dryWorldGiBodyMask:u32=0xffffffffu;

fn dryConfiguredMapping()->SvoMapping{
  var mapping=dry.mapping;
  mapping.maxVisits=min(mapping.maxVisits,dryDiagnosticMaximumNodeVisits());
  return mapping;
}
fn dryBoundThickGlassOwner(owner:u32)->bool{
  if(dryThickGlassEnabled==0u){return false;}let count=min(thickGlass.metadata.x,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u);
  for(var index=0u;index<${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u;index+=1u){if(index>=count){break;}if(svoThickGlassOwnerId(thickGlass.records[index])==owner){return true;}}
  return false;
}
// Authored records the analytic passes must not draw: the one the editor is
// previewing, and any record a thick-glass volume has taken over. The visibility
// ray's own surface used to be a third term and is gone — see
// dryVisibilityIgnoredBody for why it was never the same kind of number.
fn dryOpaqueOwnerSuppressed(owner:u32)->bool{return owner==dry.metadata.z||dryBoundThickGlassOwner(owner);}

fn missHit()->DryHit { return DryHit(DRY_MISS,vec3f(0.0,1.0,0.0),0u,DRY_OWNER_NONE,SVO_FEATURE_SMOOTH,0u,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u)); }
${screenSpaceProxyWGSL}
fn qrotWxyz(q:vec4f,v:vec3f)->vec3f { let a=cross(q.yzw,v); return v+2.0*(q.x*a+cross(q.yzw,a)); }
fn qinvWxyz(q:vec4f,v:vec3f)->vec3f { return qrotWxyz(vec4f(q.x,-q.yzw),v); }

fn slabHit(ro:vec3f,rd:vec3f,extent:vec3f)->vec2f {
  let inverse=1.0/rd; let first=(-extent-ro)*inverse; let second=(extent-ro)*inverse;
  let near=min(first,second); let far=max(first,second);
  return vec2f(max(max(near.x,near.y),near.z),min(min(far.x,far.y),far.z));
}

struct DryBoundsInterval{nearT:f32,farT:f32,valid:u32}
fn dryBoundsInterval(minimum:vec3f,maximum:vec3f,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryBoundsInterval{
  var nearT=tMin;var farT=tMax;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(rd[axis])<=1e-9){if(ro[axis]<minimum[axis]||ro[axis]>maximum[axis]){return DryBoundsInterval(nearT,farT,0u);}}
    else{let first=(minimum[axis]-ro[axis])/rd[axis];let second=(maximum[axis]-ro[axis])/rd[axis];nearT=max(nearT,min(first,second));farT=min(farT,max(first,second));if(nearT>farT){return DryBoundsInterval(nearT,farT,0u);}}
  }
  return DryBoundsInterval(nearT,farT,1u);
}

fn directionalLightSceneExitDistance(position:vec3f,directionToLightIn:vec3f)->f32 {
  // dryLightSample returns a unit direction for every valid light sample.
  let directionToLight=directionToLightIn;
  let minimum=vec3f(-0.5*uniforms.container.x,0.0,-0.5*uniforms.container.z);
  let maximum=vec3f(0.5*uniforms.container.x,uniforms.container.y,0.5*uniforms.container.z);
  var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(directionToLight[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/directionToLight[axis];let second=(maximum[axis]-position[axis])/directionToLight[axis];enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}
  }
  return max(exit,0.0);
}
/**
 * Whether a shadow ray has any sparse domain to cross at all.
 *
 * A zero exit distance says the segment from this receiver to this light never
 * enters the authored container — which is only ever true of a receiver outside
 * it, because a point inside always exits somewhere. The octree holds nothing
 * out there, so nothing can stand in the light's way, and the reading is *full
 * visibility*.
 *
 * Read as full shadow instead, which is what every call site here did, it
 * switched the sun off for every surface beyond the container footprint. On the
 * garden scenes the ground is an infinite plane and the container is 1.8 m
 * across, so the frame came back with a hard-edged lit rectangle sitting in a
 * dark field — legible as a floating slab, and not as the lighting bug it was.
 */
fn dryDirectionalRayLeavesDomain(maximumDistance:f32)->bool{return !(maximumDistance>0.0);}

// The analytic ground is gone, and with it the DryTerrainGrid decoder, the
// Lipschitz heightfield march, the eight-feature closed form and the surface
// normal taken from finite differences of the column heights.
//
// The heightfield is still authored and still voxelised — it is the largest
// single voxel population in the hero lane — so what was deleted is a duplicate
// surface, not the ground. Editor hover resolves authored terrain separately;
// that document query is not part of this renderer.

fn bodyHit(ro:vec3f,rd:vec3f,body:BodyGPU)->DryHit {
  let localOrigin=qinvWxyz(body.orientation,ro-body.positionRadius.xyz);
  let localDirection=qinvWxyz(body.orientation,rd);
  let shape=i32(round(body.halfSizeShape.w)); var t=DRY_MISS; var normal=vec3f(0.0,1.0,0.0);var featureId=SVO_FEATURE_SMOOTH;
  if (shape==0) {
    let radius=body.halfSizeShape.x; let b=dot(localOrigin,localDirection); let discriminant=b*b-dot(localOrigin,localOrigin)+radius*radius;
    if (discriminant>=0.0) { let root=sqrt(discriminant); t=-b-root; if(t<=1e-4){t=-b+root;} if(t>1e-4){normal=normalize(localOrigin+localDirection*t);}else{t=DRY_MISS;} }
  } else if (shape==1) {
    let interval=slabHit(localOrigin,localDirection,body.halfSizeShape.xyz); t=select(interval.x,interval.y,interval.x<=1e-4);
    if(t>1e-4&&interval.x<=interval.y){let point=localOrigin+localDirection*t;let q=abs(point/max(body.halfSizeShape.xyz,vec3f(1e-6)));if(q.x>=q.y&&q.x>=q.z){normal=vec3f(sign(point.x),0,0);featureId=SVO_FEATURE_BOX_X;}else if(q.y>=q.z){normal=vec3f(0,sign(point.y),0);featureId=SVO_FEATURE_BOX_Y;}else{normal=vec3f(0,0,sign(point.z));featureId=SVO_FEATURE_BOX_Z;}}else{t=DRY_MISS;}
  } else {
    let radius=body.halfSizeShape.x; let halfHeight=body.halfSizeShape.y; let a=dot(localDirection.xz,localDirection.xz); let b=dot(localOrigin.xz,localDirection.xz); let c=dot(localOrigin.xz,localOrigin.xz)-radius*radius;
    if(a>1e-7&&b*b-a*c>=0.0){let root=sqrt(b*b-a*c);for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let candidate=(-b+select(-root,root,rootIndex!=0u))/a;let y=localOrigin.y+localDirection.y*candidate;if(candidate>1e-4&&candidate<t&&abs(y)<=halfHeight){t=candidate;let p=localOrigin+localDirection*t;normal=normalize(vec3f(p.x,0,p.z));featureId=select(SVO_FEATURE_CYLINDER_SIDE,SVO_FEATURE_SMOOTH,shape==2);}}}
    if(shape==2){for(var side=-1.0;side<=1.0;side+=2.0){let center=vec3f(0.0,side*halfHeight,0.0);let offset=localOrigin-center;let hb=dot(offset,localDirection);let disc=hb*hb-dot(offset,offset)+radius*radius;if(disc>=0.0){let root=sqrt(disc);for(var rootIndex=0u;rootIndex<2u;rootIndex+=1u){let candidate=-hb+select(-root,root,rootIndex!=0u);if(candidate>1e-4&&candidate<t){t=candidate;normal=normalize(offset+localDirection*t);}}}}}
    else if(abs(localDirection.y)>1e-7){for(var side=-1.0;side<=1.0;side+=2.0){let candidate=(side*halfHeight-localOrigin.y)/localDirection.y;let p=localOrigin+localDirection*candidate;if(candidate>1e-4&&candidate<t&&dot(p.xz,p.xz)<=radius*radius){t=candidate;normal=vec3f(0,side,0);featureId=SVO_FEATURE_CYLINDER_CAP;}}}
  }
  return DryHit(t,qrotWxyz(body.orientation,normal),0u,DRY_OWNER_NONE,featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_RIGID,0u,body.colorSelected.w,vec3u(0u));
}

fn bodyCandidateVisible(ro:vec3f,rd:vec3f,body:BodyGPU,tMin:f32,tMax:f32)->bool{
  let localOrigin=qinvWxyz(body.orientation,ro-body.positionRadius.xyz);let localDirection=qinvWxyz(body.orientation,rd);let shape=i32(round(body.halfSizeShape.w));
  let radius=body.halfSizeShape.x;var extent=body.halfSizeShape.xyz;
  if(shape==0){extent=vec3f(radius);}else if(shape==2){extent=vec3f(radius,body.halfSizeShape.y+radius,radius);}else if(shape==3){extent=vec3f(radius,body.halfSizeShape.y,radius);}
  return dryBoundsInterval(-extent,extent,localOrigin,localDirection,tMin,tMax).valid!=0u;
}

// The same segment/sphere test bodyBoundingSphereVisible performs, against one
// published sphere enclosing every body rather than against a body. It is the
// guard that keeps shadow and contact rays from reading the body array at all,
// so it must stay conservative: a negative radius means the scene has no bodies,
// and the epsilon matches the per-body test so the two can never disagree about
// a grazing ray.
fn svoRigidBoundsIntersect(ro:vec3f,rd:vec3f,tMax:f32)->bool{
  let radius=dry.rigidBounds.w;
  if(radius<0.0){return false;}
  let centre=dry.rigidBounds.xyz;let offset=centre-ro;let projected=clamp(dot(offset,rd),0.0,tMax);
  let closest=ro+rd*projected;let bound=radius+1e-5;
  return dot(closest-centre,closest-centre)<=bound*bound;
}

fn bodyBoundingSphereVisible(ro:vec3f,rd:vec3f,body:BodyGPU,tMin:f32,tMax:f32)->bool{
  let offset=body.positionRadius.xyz-ro;let projected=clamp(dot(offset,rd),tMin,tMax);let closest=ro+rd*projected;let radius=max(body.positionRadius.w,0.0)+1e-5;
  return dot(closest-body.positionRadius.xyz,closest-body.positionRadius.xyz)<=radius*radius;
}

fn nearestBodyMaskIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32,bodyMask:u32)->DryHit {
  var best=missHit(); for(var index=0u;index<12u;index+=1u){if(index>=u32(round(uniforms.options.z))){break;}if(index==ignoredOwner||(bodyMask&(1u<<index))==0u){continue;}let body=bodies[index];if(!bodyBoundingSphereVisible(ro,rd,body,0.0,best.t)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ro,rd,body,0.0,best.t)){continue;}let hit=bodyHit(ro,rd,body);if(hit.t<best.t){best=hit;best.materialId=0x80000000u|index;best.ownerId=index;}} return best;
}
fn nearestBodyIgnoring(ro:vec3f,rd:vec3f,ignoredOwner:u32)->DryHit{return nearestBodyMaskIgnoring(ro,rd,ignoredOwner,0xffffffffu);}
${prepassBodyBlockerWGSL}
fn nearestBody(ro:vec3f,rd:vec3f)->DryHit{return nearestBodyIgnoring(ro,rd,DRY_OWNER_NONE);}

fn primitiveHit(record:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit {
  // Use the shared analytic ray contract directly. In particular, do not call
  // the bounded closest-point distance evaluator merely to recover an
  // ellipsoid normal after the ray quadratic has already found the surface.
  let exact=svoIntersectPrimitiveExact(record,ro,rd,max(tMin,1e-4),tMax,dryClusterPacking(record));
  if(exact.status!=SVO_PRIMITIVE_RAY_HIT){return missHit();}
  return DryHit(exact.t_m,exact.normal.xyz,svoPrimitiveMaterialId(record),svoPrimitiveOwnerId(record),exact.featureId,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}

// The patch half of a planar terminal: one exact finite slab out of the
// immutable structural catalogue, never the 8^3 DDA, and no source primitive
// ordering participating in a hit.
//
// Split out because the wrapper below resolves a *leaf index* into a patch
// index, and the primary walk has already done that resolution itself — it has
// to, to decide whether the payload lifecycle gate applies at all
// (\`dryPrimaryLeafResolve\`). Handing the patch index straight in is what stops
// the same node and leaf being read again to recover it.
fn dryPlanarPatchHit(ro:vec3f,rd:vec3f,patchIndex:u32,tEnter:f32,tExit:f32)->DryHit{
  if(dry.planarBoundaries.y==0u||dry.planarBoundaries.z!=${PLANAR_BOUNDARY_PATCH_BYTES}u
    ||patchIndex>=dry.planarBoundaries.x||patchIndex>=arrayLength(&dryPlanarBoundaries)){return missHit();}
  let boundary=dryPlanarBoundaries[patchIndex];
  let exact=intersectPlanarBoundary(boundary,ro,rd,max(tEnter,1e-4),tExit);
  if(exact.valid==0u){return missHit();}
  let identity=planarBoundaryIdentity(boundary);
  return DryHit(exact.tHit,exact.normal,identity&0xffffu,identity>>16u,
    SVO_FEATURE_BOX_X+exact.featureAxis,DRY_GBUFFER_FIELD_ANALYTIC,
    DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}
// Geometry and leaf index are accepted together: the node's leaf link, the
// leaf's back-reference and its terminal kind all have to agree before the
// patch above is allowed to answer.
fn dryPlanarTerminalHit(ro:vec3f,rd:vec3f,nodeIndex:u32,tEnter:f32,tExit:f32)->DryHit{
  if(nodeIndex>=svoControlLoad(0u)){return missHit();}
  let node=svoNodeLoad(nodeIndex);let leafIndex=node.links.z;
  if(leafIndex==SVO_INVALID||leafIndex>=svoControlLoad(1u)){return missHit();}
  let leaf=svoLeafLoad(leafIndex);
  if(leaf.topology.x!=nodeIndex||leaf.topology.z!=SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){return missHit();}
  return dryPlanarPatchHit(ro,rd,leaf.topology.w,tEnter,tExit);
}

// Exact planar geometry is a first-class structural visibility set, not a
// second copy of the voxel payload. Testing the compact catalogue globally is
// what keeps mixed leaves correct: their brick contains only non-planar
// residual geometry, while every admitted plane remains visible regardless of
// which side of a leaf boundary contains its finite slab.
fn dryPlanarCatalogHit(ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit{
  var best=missHit();best.t=tMax;
  if(dry.planarBoundaries.y==0u
    ||dry.planarBoundaries.z!=${PLANAR_BOUNDARY_PATCH_BYTES}u){return best;}
  let count=min(dry.planarBoundaries.x,arrayLength(&dryPlanarBoundaries));
  for(var boundaryIndex=0u;boundaryIndex<count;boundaryIndex+=1u){
    ${primaryWorkMap ? "dryPrimaryWorkPlanarTests+=1u;" : ""}
    let boundary=dryPlanarBoundaries[boundaryIndex];
    let exact=intersectPlanarBoundary(boundary,ro,rd,max(tMin,1e-4),best.t);
    if(exact.valid==0u){continue;}
    let identity=planarBoundaryIdentity(boundary);
    let owner=identity>>16u;
    if(dryOpaqueOwnerSuppressed(owner)){continue;}
    best=DryHit(exact.tHit,exact.normal,identity&0xffffu,owner,
      SVO_FEATURE_BOX_X+exact.featureAxis,DRY_GBUFFER_FIELD_ANALYTIC,
      DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
  }
  return best;
}

// Exact live-scene acceleration. Candidate nodes occupy the same fixed arena
// as primitives, after the primitive span, so this adds no storage binding.
fn traceScenePrimitives(ro:vec3f,rd:vec3f,tMin:f32,tMax:f32,ignoredOwner:u32)->DryHit{
  var best=missHit();best.t=tMax;
  if(dry.primitiveCandidates.w==0u||dry.primitiveCandidates.y==0u){return best;}
  var stack:array<u32,${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}>;
  var stackSize=1u;stack[0]=dry.primitiveCandidates.z;
  for(var visit=0u;visit<${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES}u&&stackSize>0u;visit+=1u){
    stackSize-=1u;let nodeIndex=stack[stackSize];
    if(nodeIndex>=dry.primitiveCandidates.y){continue;}
    ${primaryWorkMap ? "dryPrimaryWorkCandidateNodes+=1u;" : ""}
    let node=dryPrimitive(dry.primitiveCandidates.x+nodeIndex);
    let interval=dryBoundsInterval(bitcast<vec3f>(node.centerKind.xyz),bitcast<vec3f>(node.dimensionsIdentity.xyz),ro,rd,tMin,best.t);
    if(interval.valid==0u){continue;}
    let leftOrPrimitive=node.centerKind.w;let right=node.dimensionsIdentity.w;
    if(right==0xffffffffu){
      if(leftOrPrimitive>=dry.metadata.x){continue;}let record=dryPrimitive(leftOrPrimitive);let owner=svoPrimitiveOwnerId(record);
      if(owner==ignoredOwner||dryOpaqueOwnerSuppressed(owner)){continue;}
      ${primaryWorkMap ? "dryPrimaryWorkPrimitiveTests+=1u;" : ""}
      let candidate=primitiveHit(record,ro,rd,tMin,best.t);if(candidate.t<best.t){best=candidate;}
    }else if(stackSize+2u<=${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK}u){
      stack[stackSize]=right;stack[stackSize+1u]=leftOrPrimitive;stackSize+=2u;
    }
  }
  return best;
}

// The outward normal of the face a point sits on, for an axis-aligned cell.
//
// Called with the DDA's entry point, which lies exactly on one face, so the
// axis whose distance to a bound is zero is the face the ray came in through
// and wins the minimum outright. A point at a cell corner ties between two
// legal faces and either answer is a real face of a real cell.
//
// Distinct from dryPrimaryProxyNormal, which answers the same way but is only
// compiled when screen-space termination is on; this one is always here because
// the voxels-only primary is always here.
fn dryVoxelFaceNormal(bounds:mat2x3f,point:vec3f)->vec3f{
  let faceDistance=min(abs(point-bounds[0]),abs(bounds[1]-point));var axis=0u;
  if(faceDistance.y<faceDistance.x){axis=1u;}if(faceDistance.z<faceDistance[axis]){axis=2u;}
  var normal=vec3f(0.0);
  normal[axis]=select(-1.0,1.0,abs(point[axis]-bounds[1][axis])<abs(point[axis]-bounds[0][axis]));
  return normal;
}
${surfaceReconstructionWGSL}
fn traceLeafVoxelPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit${visitNodeParameterWGSL})->DryHit {
  ${primaryBrickSetupWGSL}
  ${primaryLodStrideWGSL}
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0)); let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9)); let deltaT=select(vec3f(DRY_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  // The leaf's identity storage, resolved once before the walk — on \`dense\` only.
  //
  // This is where the banded payload is paid for or lost. Per cell,
  // \`sceneIdentityAt\` is an occupancy word, three header words and two blob loads
  // against the flat lane's single load, so a naive substitution here is a
  // regression. Under the mask arms the per-cell *question* is \`bandedOccupied\`,
  // which needs no header at all, so there is nothing for this line to hoist and
  // the identity is resolved inside the gate instead — once per ray rather than
  // once per leaf crossed.
  ${cellIdentitySourceWGSL("hit.voxelOffset")}
  // One load a cell, and it answers everything.
  //
  // This walk used to resolve an *owner* here as well and hand it to a run
  // tracker, so that a run of cells naming one authored record could be promoted
  // to that record's exact sphere-traced surface. The promotion is gone, and
  // with it the whole run-merge, the upgrade budget, and the owner-range and
  // suppression tests that gated it: a solid voxel now carries the surface
  // orientation the promotion existed to recover, baked at voxelisation time.
  //
  // Solidity is exactly \`material != 0\`, which is what the voxeliser writes and
  // what the raw inspection lane already tested. Air is
  // \`packMaterialOwner(0, SPARSE_BRICK_NO_OWNER)\` and an unvoxelised brick is all
  // zeroes; both have a zero material, so one test covers both.
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<cellMinimum)||any(cell>cellMaximum)||entry>${primaryBrickExitWGSL}){break;}
    ${primaryMacroSkipWGSL}
    ${primaryWorkMap ? "dryPrimaryWorkVoxelCells+=1u;" : ""}
    var cellIdentity=0u;var cellSolid=false;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex<dryVoxelCapacity()){
      ${cellSolidGateWGSL("payloadIndex", "cellSolid=true;cellIdentity=identity;")}
    }
    ${primaryVoxelSurfaceWGSL}
    let advance=min(nextT.x,min(nextT.y,nextT.z)); if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return missHit();
}
// The terminal-kind test, for callers holding nothing but a leaf reference.
//
// The raster arms enter here with a synthesised hit and no records in hand, so
// this pair of loads is theirs to pay. The primary walk resolves the same two
// records once per visit and calls the halves directly.
fn traceLeafPayload(ro:vec3f,rd:vec3f,hit:SvoTraversalHit)->DryHit {
  let terminalNode=svoNodeLoad(hit.nodeIndex);
  if(terminalNode.links.z!=SVO_INVALID){let terminalLeaf=svoLeafLoad(terminalNode.links.z);
    if(terminalLeaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){
      return dryPlanarTerminalHit(ro,rd,hit.nodeIndex,hit.tEnter,hit.tExit);
    }
  }
  return traceLeafVoxelPayload(ro,rd,hit${primaryVisitRecords ? ",terminalNode" : ""});
}
${macroHddaPrimaryWGSL}

// One node and one leaf record per primary leaf visit.
//
// The cursor reads the node and leaf to report the hit (webgpu-svo-traversal.ts),
// and that read stays. What followed it was three more reads of the same pair:
// \`dryLeafCurrent\` for the structural-planar test — which short-circuits before
// the lifecycle word on a planar leaf, so the node is not read a fourth time —
// \`traceLeafPayload\` to decide the terminal kind, and \`dryPlanarTerminalHit\`
// to recover the patch index. Three further pairs, strictly dependent, on the
// path that dominates a flat scene. This collapses them to one and answers all
// three questions from registers.
//
// The three outcomes reproduce the old pair exactly:
//   - a structurally valid planar terminal bypasses the payload lifecycle
//     gate, because it owns no voxel payload for the voxeliser to stamp;
//   - a leaf whose terminal word says planar but whose node/leaf link does not
//     validate resolved to \`missHit()\` under both gates before, i.e. skipped;
//   - anything else is a voxel brick, admitted on the live publication gate.
const DRY_LEAF_SKIP:u32=0u;
const DRY_LEAF_VOXELS:u32=1u;
const DRY_LEAF_PLANAR:u32=2u;
struct DryLeafResolution{kind:u32,patchIndex:u32}
${primaryLeafResolveSignatureWGSL}
  let leafIndex=node.links.z;
  if(leafIndex!=SVO_INVALID){
    ${primaryLeafResolveLeafLoadWGSL}
    if(leaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){
      if(nodeIndex>=svoControlLoad(0u)||leafIndex>=svoControlLoad(1u)
        ||leaf.topology.x!=nodeIndex){return DryLeafResolution(DRY_LEAF_SKIP,0u);}
      return DryLeafResolution(DRY_LEAF_PLANAR,leaf.topology.w);
    }
  }
  if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return DryLeafResolution(DRY_LEAF_SKIP,0u);}
  return DryLeafResolution(DRY_LEAF_VOXELS,0u);
}

${primaryEntrySeedLibraryWGSL}
// The voxel-resolved surface first, the analytic set bounded by it second.
//
// Both tiers answer the same question and the frame keeps whichever is nearer,
// so the order between them is free to choose — and it is not free to get
// wrong. The analytic tier is one candidate-BVH walk whose node rejection is
// dryBoundsInterval(..., tMin, best.t): seeded with DRY_MISS it descends every
// box the ray touches anywhere along its whole length, which costs one interval
// test per record standing behind the first surface and is exactly the term
// that made this path scale with authored record count. Seeded with the voxel
// hit it descends only the boxes in front of a surface already found.
//
// The image is unchanged by construction: the walk still reports the nearest
// analytic hit that is nearer than the voxel one, and a walk that finds nothing
// returns its own tMax sentinel, which the comparison below reads as "no nearer
// analytic surface" rather than as a hit.
//
// Measured on the hero at 800x460, cone scale 0.5, all arms interleaved in one
// process (serialized submit-to-fence): 292.5 -> 222.0 ms at 501 records and
// 1564.4 -> 546.3 ms at 5 039. It is the single largest term in the 10x gap.
fn traceStaticFrom(ro:vec3f,rd:vec3f,initialMinimum:f32)->DryHit {
  // An unpublished scene has no voxels, and voxels are the only surface. Drawing
  // it analytically here would hide exactly the failure worth seeing: a frame
  // that looks perfect because the voxel path never ran. Miss instead and let
  // the existing publication tripwire say why.
  if(dryPublicationWord(0u)==0u||(dryPublicationWord(1u)&REQUIRED_FIELDS)!=REQUIRED_FIELDS){return missHit();}
  ${primaryEntrySeedTakeWGSL}
  var seeded=dryPlanarCatalogHit(ro,rd,initialMinimum,DRY_MISS);
  ${analyticPrimaryUnbounded ? "let primitiveBest=traceScenePrimitives(ro,rd,initialMinimum,seeded.t,DRY_OWNER_NONE);if(primitiveBest.t<seeded.t){seeded=primitiveBest;}" : ""}
  ${primaryEntrySeedResolveWGSL}
  var voxel=missHit();
  var minimum=max(initialMinimum,0.0);
  ${primaryEntrySeedMinimumWGSL}
  // The hierarchy is walked in front of a surface that is already known.
  //
  // \`seeded\` is the exact nearest hit of the structural planar catalogue, and
  // nothing behind it can be drawn: the return below picks the voxel only when
  // it is nearer. The cursor nevertheless used to run to \`DRY_MISS\`, so every
  // node and leaf under the stage floor was still expanded, interval-tested and
  // visited before the root finally exited — on a flat scene, most of the walk.
  //
  // Clamping the cursor's far bound to the seed deletes exactly that tail.
  // \`svoRayAabbWithInverse\` is inclusive at \`tMax\`, so the leaf holding the
  // seeded surface is still reached and any voxel in front of it is still
  // found; with nothing seeded the bound is \`DRY_MISS\`, i.e. unchanged.
  let traversalMaximum=seeded.t;
  let mapping=dryConfiguredMapping();
  let leafBudget=clamp(dry.tuningCounts0.x,1u,${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u);
  var continuation:DryTraversalCursor;
  var traversalFinished=false;
  dryTraversalCursorBegin(SvoRay(ro,minimum,rd,traversalMaximum),mapping,&continuation);
  for(var leafVisit=0u;leafVisit<${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u&&leafVisit<leafBudget;leafVisit+=1u){
    let ray=SvoRay(ro,minimum,rd,traversalMaximum);
    let leaf=dryTraversalCursorNextPrimary(ray,mapping,&continuation);
    ${primaryWorkMap ? "dryPrimaryWorkNodeVisits+=leaf.visits;" : ""}

    ${screenSpaceProxyTraceWGSL}
    if(leaf.status!=SVO_STATUS_HIT){
      traversalFinished=true;
      if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){}
      else if(leaf.status!=SVO_STATUS_MISS){}
      break;
    }
    ${primaryWorkMap ? "dryPrimaryWorkLeafVisits+=1u;" : ""}
    ${primaryVisitRecordTakeWGSL}
    let resolved=${primaryLeafResolveCallWGSL};
    if(resolved.kind==DRY_LEAF_SKIP){minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);continue;}
    var payloadHit:DryHit;
    if(resolved.kind==DRY_LEAF_PLANAR){payloadHit=dryPlanarPatchHit(ro,rd,resolved.patchIndex,leaf.tEnter,leaf.tExit);}
    else{payloadHit=${primaryLeafVoxelTraceCallWGSL}(ro,rd,leaf${primaryVisitRecords ? ",visitNode" : ""});}
    // Leaves partition space and are visited front to back, so the first payload
    // hit is the nearest voxel-resolved surface and no later leaf can beat it.
    if(payloadHit.t<seeded.t){voxel=payloadHit;traversalFinished=true;break;}
    // A hit that is not nearer says the seed is the answer, so stop here too.
    //
    // The case this catches is a planar terminal re-reporting the very
    // catalogue record that seeded the ray. \`intersectPlanarBoundary\` derives
    // the slab's \`enter\`/\`exit\` from the ray and the record alone and then
    // picks between them on \`enter >= tMin\`; the catalogue's tMin is the ray's
    // own start and the terminal's is the leaf entry, so wherever the ray meets
    // the slab from outside the leaf both take the \`enter\` branch and report
    // bit-identical t. \`<\` alone therefore never fires on the stage floor and
    // the walk grinds on underneath it. Where the two tMins disagree — a
    // grazing ray whose slab entry precedes the leaf entry — the terminal takes
    // the \`exit\` branch instead, lands past the clamped tExit and misses, and
    // the walk ends on the next cursor call, which the clamp has already bounded
    // at the seed. Both routes return \`seeded\`.
    //
    // One arm this does move pixels in: with screen-space termination on
    // (\`screenSpaceProxyTraceWGSL\`, raster-primary/split only), a sub-pixel
    // proxy node beneath the floor used to be returned over the floor, because
    // that early return never compares against the seed. The clamp stops at the
    // seed instead. Production compiles the arm out.
    if(payloadHit.t<DRY_MISS){traversalFinished=true;break;}
    minimum=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
  // Reaching the uniform budget without an authoritative hierarchy miss is a
  // traversal exhaustion, not an empty scene. Keep that visible in the
  // existing failure heatmap instead of silently returning black.
  if(!traversalFinished){}
  if(voxel.t<seeded.t){return voxel;}return seeded;
}
fn traceStatic(ro:vec3f,rd:vec3f)->DryHit{return traceStaticFrom(ro,rd,0.0);}

struct DryGlassHit{hit:SvoThinGlassHit,recordIndex:u32}
fn dryGlassMiss()->DryGlassHit{return DryGlassHit(svoThinGlassMiss(),0u);}
fn dryGlassBoundingSphereVisible(record:SvoThinGlassRecord,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->bool{
  let offset=record.centerThickness.xyz-ro;let projected=clamp(dot(offset,rd),tMin,tMax);let closest=ro+rd*projected;let radius=length(vec3f(record.extentIorEpsilon.xy,.5*record.centerThickness.w))+record.extentIorEpsilon.w+1e-5;
  return dot(closest-record.centerThickness.xyz,closest-record.centerThickness.xyz)<=radius*radius;
}
fn traceGlass(ro:vec3f,rd:vec3f,tMin_m:f32,tMax_m:f32)->DryGlassHit {
  var best=dryGlassMiss();var bestT=tMax_m;
  let paneCount=min(dry.glass.y,${SVO_SCENE_GLASS_MAXIMUM_PANES}u);
  for(var paneIndex=0u;paneIndex<${SVO_SCENE_GLASS_MAXIMUM_PANES}u;paneIndex+=1u){
    if(paneIndex>=paneCount){break;}let record=dryGlassPane(paneIndex);let paneId=svoThinGlassPaneId(record);let thickReplaced=dryThickGlassEnabled!=0u&&paneId==thickGlass.metadata.z;if(thickReplaced||!dryGlassBoundingSphereVisible(record,ro,rd,tMin_m,bestT)){continue;}let candidate=svoThinGlassIntersect(record,ro,rd,tMin_m,bestT,1e-6,record.extentIorEpsilon.w);
    if(candidate.valid!=0u&&candidate.t_m<bestT){best=DryGlassHit(candidate,paneIndex);bestT=candidate.t_m;}
  }
  return best;
}

struct DryThickGlassHit{interval:SvoThickGlassInterval,recordIndex:u32}
fn dryThickGlassMiss()->DryThickGlassHit{return DryThickGlassHit(svoThickGlassEmpty(SVO_THICK_GLASS_MISS),0u);}
fn dryThickGlassFirst(interval:SvoThickGlassInterval)->SvoThickGlassSurface{var first=interval.exit;if(interval.hasEntry!=0u){first=interval.entry;}return first;}
fn traceThickGlass(ro:vec3f,rd:vec3f,tMin_m:f32,tMax_m:f32)->DryThickGlassHit{
  var best=dryThickGlassMiss();var bestT=tMax_m;if(dryThickGlassEnabled==0u){return best;}
  let count=min(thickGlass.metadata.x,${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u);
  for(var recordIndex=0u;recordIndex<${SVO_SCENE_THICK_GLASS_MAXIMUM_VOLUMES}u;recordIndex+=1u){
    if(recordIndex>=count){break;}let candidate=svoThickGlassIntersect(thickGlass.records[recordIndex],ro,rd,tMin_m,bestT,thickGlass.metadata.y);
    if(candidate.status==SVO_THICK_GLASS_INVALID||candidate.status==SVO_THICK_GLASS_STALE){dryThickGlassFailure=candidate.status;return dryThickGlassMiss();}
    if(candidate.status==SVO_THICK_GLASS_HIT){let first=dryThickGlassFirst(candidate);if(first.t_m<bestT){best=DryThickGlassHit(candidate,recordIndex);bestT=first.t_m;}}
  }
  return best;
}

fn dryVisibilityStep(status:u32,nodeVisits:u32,leafVisits:u32,workItems:u32,t:f32)->SvoVisibilityStep {
  return SvoVisibilityStep(status,nodeVisits,leafVisits,workItems,t,1u,vec3f(0.0),0u);
}
fn dryVisibilityTransmissionStep(nodeVisits:u32,leafVisits:u32,workItems:u32,t:f32,transmittance:vec3f)->SvoVisibilityStep {
  return SvoVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,t,0u,clamp(transmittance,vec3f(0.0),vec3f(1.0)),0u);
}
fn dryMaterialPublished(material:SvoMaterialRecord,index:u32)->bool{
  return index<dry.materialPublication.x&&svoMaterialValid(material,index)&&material.identity.y==dry.materialPublication.y;
}
fn dryMaterialThinDielectric(material:SvoMaterialRecord,index:u32)->bool{
  let required=SVO_MATERIAL_FLAG_DIELECTRIC|SVO_MATERIAL_FLAG_THIN_WALL;
  return dryMaterialPublished(material,index)&&(material.identity.w&required)==required;
}
fn dryHitThinDielectric(hit:DryHit)->bool{
  let materialId=dryResolvedMaterialId(hit);
  return materialId<dry.materialPublication.x&&dryMaterialThinDielectric(dryMaterial(materialId),materialId);
}
fn dryThinDielectricTransmittance(material:SvoMaterialRecord,normal:vec3f,direction:vec3f)->vec3f{
  let cosine=clamp(abs(dot(normalize(normal),normalize(direction))),0.0,1.0);
  let f0=svoMaterialDielectricF0(material);let fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
  let tint=mix(vec3f(1.0),clamp(material.scatteringColorAnisotropy.xyz,vec3f(0.0),vec3f(1.0)),clamp(material.baseColorOpacity.w,0.0,1.0));
  return tint*clamp(material.surface.w*(1.0-fresnel),0.0,1.0);
}
fn dryPlanarTerminalVisibility(ray:SvoVisibilityRay,hit:SvoTraversalHit)->SvoVisibilityStep{
  let candidate=dryPlanarTerminalHit(ray.origin_m,ray.direction,hit.nodeIndex,hit.tEnter,
    min(hit.tExit,ray.tMax_m));
  if(!(candidate.t<DRY_MISS)){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,1u,DRY_MISS);}
  let materialId=dryResolvedMaterialId(candidate);
  if(materialId>=dry.materialPublication.x){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,1u,DRY_MISS);}
  let material=dryMaterial(materialId);
  if(!dryMaterialPublished(material,materialId)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,1u,DRY_MISS);}
  if(dryMaterialThinDielectric(material,materialId)){
    return dryVisibilityTransmissionStep(0u,0u,1u,min(hit.tExit,ray.tMax_m),
      dryThinDielectricTransmittance(material,candidate.normal,ray.direction));
  }
  return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,1u,candidate.t);
}

// Renderer-local unit-vector variant of the shared bias contract. Surface
// normals and light/contact directions are normalized at their construction
// sites, so repeating both inverse-square-roots per visibility ray is waste.
fn dryBiasedVisibilityRayUnit(surfacePosition_m:vec3f,geometricNormal:vec3f,directionToLight:vec3f,maximumLightDistance_m:f32,cellSize_m:vec3f,biasCells:f32)->SvoVisibilityRay {
  let projectedCellWidth=dot(abs(geometricNormal),cellSize_m);let originBias_m=max(biasCells,0.0)*projectedCellWidth;
  let side=select(-1.0,1.0,dot(geometricNormal,directionToLight)>=0.0);let offset=side*geometricNormal*originBias_m;
  return SvoVisibilityRay(surfacePosition_m+offset,max(0.0,maximumLightDistance_m-dot(offset,directionToLight)),directionToLight,originBias_m);
}

// Shadow payload lookup mirrors the production leaf DDA, but reports invalid
// data and bounded-work exhaustion explicitly so direct light fails closed.
fn traceLeafPayloadVisibility(ray:SvoVisibilityRay,tMin_m:f32,hit:SvoTraversalHit,workLimit:u32)->SvoVisibilityStep {
  let terminalNode=svoNodeLoad(hit.nodeIndex);
  if(terminalNode.links.z!=SVO_INVALID){let terminalLeaf=svoLeafLoad(terminalNode.links.z);
    if(terminalLeaf.topology.z==SVO_LEAF_TERMINAL_PLANAR_BOUNDARY){
      if(workLimit==0u){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,0u,DRY_MISS);}
      return dryPlanarTerminalVisibility(ray,hit);
    }
  }
  ${shadowBrickSetupWGSL}
  let step=select(vec3i(-1),vec3i(1),ray.direction>=vec3f(0.0));let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(DRY_MISS),(nextBoundary-ray.origin_m)/ray.direction,abs(ray.direction)>vec3f(1e-9));let deltaT=select(vec3f(DRY_MISS),abs(extent/ray.direction),abs(ray.direction)>vec3f(1e-9));
  var workItems=0u;
  // A solid voxel is an occluder, and this walk is where the frame says so.
  //
  // It used to step cells, resolve an owner and drop it, so it never produced a
  // hit at all and every shadow came from the analytic candidate BVH beside it.
  // That is why the disabled-analytic probe measured *slower* than either real
  // arm: it deleted the occluders rather than the work, and unshadowed rays then
  // ran to full tMax. Reading the same solidity the primary reads — material
  // != 0, one gate, no owner — is what makes deleting the analytic tier a
  // deletion instead of an image change.
  //
  // The distance reported is \`entry\`, the ray's own distance to the face it
  // crossed to reach this cell, so a transmittance term integrated along the ray
  // stops exactly where the primary would have drawn the surface.
  ${cellIdentitySourceWGSL("hit.voxelOffset")}
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<cellMinimum)||any(cell>cellMaximum)||entry>${shadowBrickExitWGSL}||entry>ray.tMax_m){return dryVisibilityStep(SVO_VIS_STEP_MISS,0u,0u,workItems,DRY_MISS);}
    ${shadowMacroSkipWGSL}if(workItems>=workLimit){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);}workItems+=1u;
    let payloadIndex=svoBrickVoxelIndex(hit.voxelOffset,vec3u(cell),dry.mapping.brickSize);
    if(payloadIndex>=dryVoxelCapacity()){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}
    ${cellSolidGateWGSL("payloadIndex", "let materialId=sceneIdentityMaterial(identity);if(materialId>=dry.materialPublication.x){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}let material=dryMaterial(materialId);if(!dryMaterialPublished(material,materialId)){return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,workItems,DRY_MISS);}if(dryMaterialThinDielectric(material,materialId)){let cellBounds=mat2x3f(bounds[0]+vec3f(cell)*extent,bounds[0]+(vec3f(cell)+vec3f(1.0))*extent);let normal=dryVoxelFaceNormal(cellBounds,ray.origin_m+ray.direction*entry);let cellExit=min(nextT.x,min(nextT.y,nextT.z));return dryVisibilityTransmissionStep(0u,0u,workItems,min(max(cellExit,entry),ray.tMax_m),dryThinDielectricTransmittance(material,normal,ray.direction));}return dryVisibilityStep(SVO_VIS_STEP_HIT,0u,0u,workItems,entry);")}
    let advance=min(nextT.x,min(nextT.y,nextT.z));if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}entry=advance;
  }
  return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,0u,0u,workItems,DRY_MISS);
}
${macroHddaShadowWGSL}

const DRY_MEDIUM_AIR:u32=0u;
fn dryThinGlassIncidentIor()->f32{return 1.0;}

// Adapter required by svoTraceVisibility. It returns the nearest opaque or
// transmissive candidate and never calls the lighting closure recursively.
fn svoVisibilityNext(ray:SvoVisibilityRay,tMin_m:f32,remaining:SvoVisibilityBudget)->SvoVisibilityStep {
  if(dryPublicationWord(0u)==0u||(dryPublicationWord(1u)&REQUIRED_FIELDS)!=REQUIRED_FIELDS){dryVisibilityStepInvalidReason=1u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  if(dry.glass.y>${SVO_SCENE_GLASS_MAXIMUM_PANES}u){dryVisibilityStepInvalidReason=2u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,0u,0u,0u,DRY_MISS);}
  var nodeVisits=0u;var leafVisits=0u;var workItems=0u;var bestT=ray.tMax_m;var found=false;var opaque=true;var glassTransmission=vec3f(0.0);

  let bodyCount=min(u32(round(max(uniforms.options.z,0.0))),12u);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}if(bodyIndex==dryVisibilityIgnoredBody){continue;}if(workItems>=remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=1u;
    let body=bodies[bodyIndex];if(!bodyBoundingSphereVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let shape=i32(round(body.halfSizeShape.w));if(shape>=2&&!bodyCandidateVisible(ray.origin_m,ray.direction,body,tMin_m,bestT)){continue;}let candidate=bodyHit(ray.origin_m,ray.direction,body);if(candidate.t>=tMin_m&&candidate.t<bestT){bestT=candidate.t;found=true;opaque=true;}
  }

  // The voxel-resolved tier, and there is no second tier behind it.
  //
  // Every shadow, AO and GI ray used to open with a candidate-BVH walk over the
  // authored records — bounded, for a shadow ray, by the whole distance to the
  // light — so each one paid an interval test per record standing anywhere along
  // it, and that walk supplied *all* of this path's occluders because the leaf
  // DDA beside it resolved an owner and dropped it. Both halves of that are gone:
  // the leaf walk tests the same solidity the primary tests, so the analytic set
  // has nothing left to contribute that the voxels do not already hold.
  //
  // Ordering is no longer a question, and neither is exhaustion. A walk that runs
  // out of budget returns EXHAUSTED and the caller fails the ray closed, exactly
  // as it did when the analytic reach could not rescue it.
  var cursor=max(tMin_m,0.0);var shadowContinuation:DryTraversalCursor;let initialShadowMapping=dryConfiguredMapping();dryTraversalCursorBegin(SvoRay(ray.origin_m,cursor,ray.direction,bestT),initialShadowMapping,&shadowContinuation);
  for(var leafAttempt=0u;leafAttempt<${SVO_VISIBILITY_LIMITS.leafVisits}u;leafAttempt+=1u){
    if(cursor>=bestT){break;}if(leafVisits>=remaining.leafVisits||nodeVisits>=remaining.nodeVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
    var shadowMapping=dryConfiguredMapping();shadowMapping.maxVisits=min(shadowMapping.maxVisits,remaining.nodeVisits-nodeVisits);
    let leaf=dryTraversalCursorNext(SvoRay(ray.origin_m,cursor,ray.direction,bestT),shadowMapping,&shadowContinuation);nodeVisits+=leaf.visits;
    if(leaf.status==SVO_STATUS_MISS){break;}
    if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}
    if(leaf.status!=SVO_STATUS_HIT){dryVisibilityStepInvalidReason=3u;return dryVisibilityStep(SVO_VIS_STEP_INVALID,nodeVisits,leafVisits,workItems,DRY_MISS);}leafVisits+=1u;
    if(!dryLeafCurrent(leaf)){cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);continue;}
    let payloadRay=SvoVisibilityRay(ray.origin_m,bestT,ray.direction,ray.originBias_m);let payload=${shadowLeafTraceCallWGSL}(payloadRay,tMin_m,leaf,remaining.workItems-workItems);workItems+=payload.workItems;
    // Leaves partition space and are visited front to back, so the first payload
    // hit is the nearest voxel occluder and no later leaf can beat it.
    if(payload.status==SVO_VIS_STEP_HIT){bestT=payload.t_m;found=true;opaque=payload.opaque!=0u;glassTransmission=payload.transmittance;break;}
    if(payload.status!=SVO_VIS_STEP_MISS){if(payload.status==SVO_VIS_STEP_INVALID){dryVisibilityStepInvalidReason=3u;}return dryVisibilityStep(payload.status,nodeVisits,leafVisits,workItems,payload.t_m);}
    cursor=leaf.tExit+max(1e-5,length(dry.mapping.cellSize)*1e-3);
  }
  if(cursor<bestT&&leafVisits>=remaining.leafVisits){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}

  let paneCount=dry.glass.y;if(workItems+paneCount>remaining.workItems){return dryVisibilityStep(SVO_VIS_STEP_EXHAUSTED,nodeVisits,leafVisits,workItems,DRY_MISS);}workItems+=paneCount;
  // Authored environment glazing remains a finite-pane query. SolidWorld glass
  // was already resolved above from the voxel's published material record.
  let glass=traceGlass(ray.origin_m,ray.direction,tMin_m,bestT);if(glass.hit.valid!=0u&&glass.hit.t_m<bestT){let optics=svoThinGlassOptics(dryGlassPane(glass.recordIndex),glass.hit,dryThinGlassIncidentIor());bestT=glass.hit.t_m;found=true;opaque=false;glassTransmission=optics.netTransmittance;}
  if(!found){return dryVisibilityStep(SVO_VIS_STEP_MISS,nodeVisits,leafVisits,workItems,DRY_MISS);}if(opaque){return dryVisibilityStep(SVO_VIS_STEP_HIT,nodeVisits,leafVisits,workItems,bestT);}return dryVisibilityTransmissionStep(nodeVisits,leafVisits,workItems,bestT,glassTransmission);
}

${svoVisibilityTraceWGSL}

// Water attenuates by wavelength over distance instead of blocking: red is gone
// long before blue-green is. Extinction, not absorption: light scattered out of
// the beam has left it just as surely as light absorbed, and for water the
// scattering coefficient is a third of extinction in blue-green while being
// almost nothing in red, so dropping it both weakens the shadow and warms it.
fn dryFluidTransmittance(depth_m:f32)->vec3f {
  if(!(depth_m>0.0)){return vec3f(1.0);}
  return unifiedBeerLambert(vec3f(${WATER_OPTICS.absorption.map((value, channel) => value + WATER_OPTICS.scatter[channel]!).join(",")}),depth_m);
}

// Entry and exit distance along a ray through the coverage volume's box.
// Outside it coverage is defined to be zero, so an unclipped march spends its
// whole step budget on empty space — and, for a receiver far from the water,
// steps clean past the liquid without ever sampling it.
fn dryFluidCoverageSlab(origin:vec3f,direction:vec3f,maximumDistance_m:f32)->vec2f {
  let lower=dry.fluidCoverage.origin_m;
  let upper=lower+vec3f(dry.fluidCoverage.dimensions)*dry.fluidCoverage.texelSize_m;
  // An axis-parallel ray gets a denormal-free infinity rather than a branch: the
  // resulting +/-inf slab either fails to constrain the interval or empties it,
  // which is the correct answer in both cases.
  let inverse=1.0/select(direction,vec3f(1e-20),abs(direction)<vec3f(1e-20));
  let first=(lower-origin)*inverse;let second=(upper-origin)*inverse;
  let near=min(first,second);let far=max(first,second);
  return vec2f(max(0.0,max(near.x,max(near.y,near.z))),min(maximumDistance_m,min(far.x,min(far.y,far.z))));
}

/**
 * Path length through liquid along a cone, in metres.
 *
 * Deliberately a separate march rather than a lane on the node-mip cone. The
 * node-mip pyramid is a publish-once structure over authored scenery, and every
 * cheap path built on it — the reduced-rate prepass, the voxel light cache —
 * reuses a value across pixels or frames. Water moves every frame and is
 * routinely airborne where no static page exists, so a water term carried
 * through those caches is stale exactly when it matters. This volume is instead
 * a small mipped 3D texture with hardware trilinear and linear mip filtering:
 * marching it is a handful of texture fetches with no page lookup, cheap enough
 * to run at full rate behind every cached path, which is what lets the reduced
 * and exact pixels agree about the water rather than disagreeing at the seam.
 *
 * stepWidth times coverage is the length of the step that lies inside liquid,
 * so the sum is a measured path length. No self-occlusion weight: a receiver
 * standing in a pond really is under water, and suppressing the first samples
 * would erase exactly the shading that makes it read as wet.
 */
fn dryFluidOpticalDepth(origin:vec3f,direction:vec3f,maximumDistance_m:f32,aperture:f32)->f32 {
  if(!svoFluidCoverageReady(dry.fluidCoverage)||!(maximumDistance_m>0.0)){return 0.0;}
  let texel_m=max(dry.fluidCoverage.texelSize_m.x,max(dry.fluidCoverage.texelSize_m.y,dry.fluidCoverage.texelSize_m.z));
  if(!(texel_m>0.0)){return 0.0;}
  let slab=dryFluidCoverageSlab(origin,direction,maximumDistance_m);
  if(!(slab.y>slab.x)){return 0.0;}
  let tangent=tan(aperture*.5);
  var distance=slab.x;var depth_m=0.0;
  for(var stepIndex=0u;stepIndex<${SVO_DRY_FLUID_MARCH_STEPS}u;stepIndex+=1u){
    if(distance>=slab.y){break;}
    // The cone footprint sets both the step and the mip level, so a widening
    // cone costs a logarithmic number of steps and reads progressively blurrier
    // levels — the same softening the scenery cone gets, so a water shadow and a
    // solid one cast from the same emitter have the same penumbra.
    let diameter=max(texel_m,2.0*distance*tangent);
    let stepWidth=min(diameter,slab.y-distance);
    let coverage=svoFluidCoverageAt(fluidCoverageVolume,nodeMipSampler,dry.fluidCoverage,origin+direction*(distance+stepWidth*.5),svoFluidCoverageLod(diameter,texel_m));
    depth_m+=stepWidth*coverage;
    distance+=stepWidth;
  }
  return depth_m;
}

/**
 * Direct-light visibility, water included.
 *
 * The solid term has five exits — the voxel light cache, the reduced-rate
 * prepass, the cone march, the exact SVO trace, and the early "this ray leaves
 * the scene" shortcut — and three of them are cached paths that cannot carry a
 * per-frame liquid term. Composing the water factor once, over whichever exit
 * fired, is what makes the effect seamless: a pixel served by the cache and its
 * neighbour served by the exact trace apply the identical attenuation, so the
 * water shadow has no seam where the shading path changes.
 *
 * It also shares the solid shadow's strength knob, so a scene that softens or
 * disables shadows softens or disables the water's shadow with it.
 */
fn dryLightVisibility(position:vec3f,geometricNormal:vec3f,ownerId:u32,towardLight:vec3f,finiteDistance_m:f32)->vec3f {
  let solid=dryLightVisibilitySolid(position,geometricNormal,ownerId,towardLight,finiteDistance_m);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.exactShadow}u)==0u
    ||!svoFluidCoverageReady(dry.fluidCoverage)||all(solid<=vec3f(0.0))){return solid;}
  let maximumDistance=select(directionalLightSceneExitDistance(position,towardLight),finiteDistance_m,finiteDistance_m>0.0);
  let depth_m=dryFluidOpticalDepth(position,towardLight,maximumDistance,dry.tuningRays1.y);
  return solid*mix(vec3f(1.0),dryFluidTransmittance(depth_m),dry.tuningRays0.y);
}

fn dryLightVisibilitySolid(position:vec3f,geometricNormal:vec3f,ownerId:u32,towardLight:vec3f,finiteDistance_m:f32)->vec3f {
  if(dot(geometricNormal,towardLight)<=0.0){return vec3f(0.0);}
  if((dry.materialPublication.w&2u)==0u){return vec3f(1.0);}
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}
  let maximumDistance=select(directionalLightSceneExitDistance(position,towardLight),finiteDistance_m,finiteDistance_m>0.0);if(dryDirectionalRayLeavesDomain(maximumDistance)){return vec3f(1.0);}
  let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,towardLight,maximumDistance,dry.mapping.cellSize,dry.tuningRays0.x);
  ${voxelLightCacheShortcutWGSL}
  // Direct light remains an analytic contribution from the current live light
  // arena. The derived radiance atlas contains emitted energy and supplements
  // this term; it is never a baked replacement for current scene lighting.
  // Reuse hierarchical cone visibility for the analytic direct term instead of
  // recasting a full exact SVO ray for every receiver and every authored light.
  // Reduced shading retains its full-rate analytic rigid-body correction in
  // prepassShadowShortcutWGSL. Cone mode has no undeclared exact escape: an
  // unavailable requested page publishes a typed fail-closed diagnostic.
  if(${fastDeferred ? "true" : `(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u`}){
    if(!dryNodeMipReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}${prepassShadowShortcutWGSL}
    // The cone origin escapes the receiver's own trilinear coverage support
    // along the geometric normal: the 0.02-cell hard-ray bias alone leaves the
    // first cone samples inside the surface, whose accumulated self-occlusion
    // renders as banding. Finite emitters additionally clear the march end by
    // one cone-support width: a march ending exactly at the emitter surface
    // reads the emitter's own voxelized coverage through the last samples'
    // trilinear/mip support, and the amount aliases with the receiver's
    // distance modulo the step size as concentric rings around the light.
    let coneCell_m=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
    let coneEscape_m=coneCell_m*dry.tuningRays1.z;
    let coneMaxRaw_m=max(0.0,ray.tMax_m-coneEscape_m*dot(geometricNormal,towardLight));
    let coneMax_m=coneMaxRaw_m-select(0.0,dry.tuningRays1.w*coneCell_m,finiteDistance_m>0.0);
    let cone=dryConeVisibility(ray.origin_m+geometricNormal*coneEscape_m,towardLight,dry.tuningRays1.y,coneMax_m,geometricNormal,finiteDistance_m>0.0);
    if(cone.valid==0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f(0.0);}
    let rigidBlocker=nearestBodyIgnoring(ray.origin_m,towardLight,ownerId);if(rigidBlocker.t<ray.tMax_m){return vec3f(1.0-dry.tuningRays0.y);}return mix(vec3f(1.0),vec3f(cone.transmittance),dry.tuningRays0.y);
  }
  dryVisibilityIgnoredBody=ownerId;
  let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_EXHAUSTED){}else if(result.status==SVO_VIS_STATUS_INVALID){}
  dryVisibilityIgnoredBody=DRY_OWNER_NONE;
  return mix(vec3f(1.0),result.transmittance,dry.tuningRays0.y);
}

fn dryContactVisibilityRadius()->f32 {
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let sceneScale=max(uniforms.container.x,max(uniforms.container.y,uniforms.container.z));
  return dry.tuningRays0.z*min(sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.maximumSceneRadiusFraction},max(cellScale*${SVO_CONTACT_VISIBILITY_CONTRACT.radiusCells}.0,sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.minimumSceneRadiusFraction}));
}
fn dryContactVisibilityDirection(geometricNormalIn:vec3f,featureId:u32,sampleIndex:u32)->vec3f {
  let geometricNormal=normalize(geometricNormalIn);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(geometricNormal.y)>.9);var tangent=normalize(cross(helper,geometricNormal));var bitangent=cross(geometricNormal,tangent);
  if((featureId&1u)!=0u){let previous=tangent;tangent=bitangent;bitangent=-previous;}
  let signValue=select(1.0,-1.0,sampleIndex!=0u);return normalize(geometricNormal+signValue*(.55*tangent+.2*bitangent));
}
/**
 * Contact visibility, water included.
 *
 * This is the term the eye reads as "resting on" rather than "floating above",
 * and it is the one a shadow alone cannot supply: the floor immediately beside
 * a tank is still lit by the lamp, so only the near-field hemisphere darkening
 * tells you the water is there. Sampled along the same cone directions the solid
 * term uses, so the two darken the same neighbourhood and blend rather than
 * fighting; averaged in linear transmittance, which is what the solid term
 * averages too.
 */
fn dryContactVisibility(position:vec3f,geometricNormal:vec3f,featureId:u32,ownerId:u32)->vec3f {
  let solid=dryContactVisibilitySolid(position,geometricNormal,featureId,ownerId);
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)==0u
    ||!svoFluidCoverageReady(dry.fluidCoverage)||all(solid<=vec3f(0.0))){return solid;}
  let radius=dryContactVisibilityRadius();if(radius<=0.0){return solid;}
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let origin=position+normalize(geometricNormal)*cellScale*.2;
  var transmittance=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<${SVO_DRY_CONTACT_FLUID_SAMPLES}u;sampleIndex+=1u){
    let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);
    transmittance+=dryFluidTransmittance(dryFluidOpticalDepth(origin,direction,radius,dry.tuningRays1.x));
  }
  return solid*mix(vec3f(1.0),transmittance/f32(${SVO_DRY_CONTACT_FLUID_SAMPLES}),dry.tuningRays0.w);
}

fn dryContactVisibilitySolid(position:vec3f,geometricNormal:vec3f,featureId:u32,ownerId:u32)->vec3f {
  if((dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion}u)==0u){return vec3f(1.0);}
  if((dryDerivedPageFailure&${SVO_DRY_DERIVED_FAILURE.reducedReconstruction}u)!=0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}
  if(${fastDeferred ? "true" : `(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested}u)!=0u`}){
    if(!dryNodeMipReady()){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}${prepassContactShortcutWGSL}
    let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));let origin=position+normalize(geometricNormal)*cellScale*.2;let coneSampleCount=max(dry.tuningCounts1.z,dry.tuningCounts1.y);
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_STABLE_AO_CONE_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=coneSampleCount){break;}let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex&1u);let rotated=select(direction,normalize(direction+cross(normalize(geometricNormal),direction)*.7),sampleIndex>=2u);let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);if(cone.valid==0u){dryDerivedPageFailure|=${SVO_DRY_DERIVED_FAILURE.ambientOcclusionPage}u;return vec3f(0.0);}let rigidBlocker=nearestBodyIgnoring(origin,rotated,ownerId);visibility+=select(cone.transmittance,0.0,rigidBlocker.t<radius);}let raw=clamp(visibility/f32(coneSampleCount),0.0,1.0);return vec3f(mix(1.0,raw,dry.tuningRays0.w));
  }
  if((dry.materialPublication.w&1u)==0u){return vec3f(1.0);}
  let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(0.0);}let biasCells=select(${SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells},${SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells},featureId!=SVO_FEATURE_SMOOTH);var visibility=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}u;sampleIndex+=1u){let direction=dryContactVisibilityDirection(geometricNormal,featureId,sampleIndex);let ray=dryBiasedVisibilityRayUnit(position,geometricNormal,direction,radius,dry.mapping.cellSize,biasCells);let result=svoTraceVisibility(ray,SvoVisibilityBudget(dry.tuningCounts1.w,dry.tuningCounts2.x,dry.tuningCounts2.y,dry.tuningCounts2.z),true,0.001,max(ray.originBias_m,1e-6));if(result.status==SVO_VIS_STATUS_INVALID||result.status==SVO_VIS_STATUS_EXHAUSTED){return vec3f(0.0);}visibility+=result.transmittance;}
  return mix(vec3f(1.0),clamp(visibility/f32(${SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount}),vec3f(0.0),vec3f(1.0)),dry.tuningRays0.w);
}

fn dryEnvironment(rd:vec3f,roughness:f32)->vec3f{return svoEnvironmentPrefilteredSpecular(dryLighting.environment,rd,roughness);}
struct DryLightSample{towardLight:vec3f,finiteDistance_m:f32,radiance:vec3f,valid:u32}
fn dryInvalidLightSample()->DryLightSample{return DryLightSample(vec3f(0.0,1.0,0.0),0.0,vec3f(0.0),0u);}
fn dryLightSample(light:SvoLightRecord,sampleIndex:u32,position:vec3f)->DryLightSample {
  let baseRadiance=svoLightRadiance(light);if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)<=0.0){return dryInvalidLightSample();}
  if(light.identity.x==SVO_LIGHT_DIRECTIONAL){let lengthSquared=dot(light.directionCone.xyz,light.directionCone.xyz);if(lengthSquared<=1e-12){return dryInvalidLightSample();}return DryLightSample(light.directionCone.xyz*inverseSqrt(lengthSquared),0.0,baseRadiance,1u);}
  var samplePosition=light.positionRange.xyz;
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_SPOT){
    // A spot jitters across its mouth disc for the same reason a sphere jitters
    // across its silhouette: the shadow edge of a finite emitter is a penumbra,
    // and a single centre sample renders it as a hard line.
    let towardCenter=normalize(light.positionRange.xyz-position);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(towardCenter.y)>.9);let tangent=normalize(cross(towardCenter,helper));let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=tangent*(signValue*.45*light.shape.x);
  }else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){
    let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=light.axisUWidth.xyz*(signValue*.45*light.axisUWidth.w)+light.axisVHeight.xyz*(signValue*.2*light.axisVHeight.w);
  }
  let offset=samplePosition-position;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10){return dryInvalidLightSample();}if(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w){return dryInvalidLightSample();}let distance=sqrt(distanceSquared);let towardLight=offset/distance;
  let rangeFade=select(1.0,pow(clamp(1.0-distance/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);
  var shapeScale=1.0/max(1.0,distanceSquared);
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*UNIFIED_PI*light.shape.x*light.shape.x;shapeScale=area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;let emitterFacing=max(dot(normalize(light.directionCone.xyz),-towardLight),0.0);shapeScale=emitterFacing*area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_SPOT){shapeScale*=svoLightConeFalloff(light,towardLight);}
  let radiance=baseRadiance*(rangeFade*shapeScale);if(max(max(radiance.x,radiance.y),radiance.z)<=0.0){return dryInvalidLightSample();}
  // Point fixtures retain the finite radius of their visible emissive proxy.
  // Attenuation uses center distance, while visibility ends at the globe's
  // near surface so the source geometry cannot occlude its own light.
  let visibilityDistance=select(distance,max(0.0,distance-light.shape.x),light.identity.x==SVO_LIGHT_POINT||light.identity.x==SVO_LIGHT_SPOT);
  return DryLightSample(towardLight,visibilityDistance,radiance,1u);
}
// The static scene is traceStatic alone. It used to be that plus an analytic
// terrain trace, and the wrapper that folded the two together is gone with the
// second surface: a name whose whole content is "voxels, or the ground drawn
// twice" outlives its own meaning quietly.
fn traceDrySolidSceneFrom(ro:vec3f,rd:vec3f,tMin:f32)->DryHit {
  var hit=traceStaticFrom(ro,rd,tMin);let rigid=nearestBody(ro,rd);if(rigid.t>=tMin&&rigid.t<hit.t){hit=rigid;}
  return dryPresentationHit(hit);
}
fn traceDrySolidScene(ro:vec3f,rd:vec3f)->DryHit{return traceDrySolidSceneFrom(ro,rd,0.0);}
fn traceOpaqueScene(ro:vec3f,rd:vec3f)->DryHit {
  return traceDrySolidScene(ro,rd);
}
const DRY_SURFACE_REGION_NONE:u32=0xffffffffu;
struct DrySurfaceMaterial{baseColor:vec3f,roughness:f32,emissive:vec3f,metallic:f32,specularF0:vec3f,specularWeight:f32,regionId:u32,variationFlags:u32,valid:u32,_padding:u32}
fn dryInvalidSurfaceMaterial()->DrySurfaceMaterial{return DrySurfaceMaterial(vec3f(0.0),1.0,vec3f(0.0),0.0,vec3f(0.04),0.0,DRY_SURFACE_REGION_NONE,0u,0u,0u);}
fn dryBodyPbrMaterialId(body:BodyGPU)->u32{
  let shape=i32(round(body.halfSizeShape.w));if(shape==0){return ${VOXEL_MATERIAL_IDS.sphere}u;}if(shape==1){return ${VOXEL_MATERIAL_IDS.box}u;}if(shape==2){return ${VOXEL_MATERIAL_IDS.capsule}u;}return ${VOXEL_MATERIAL_IDS.cylinder}u;
}
fn dryResolvedMaterialId(hit:DryHit)->u32{
  if((hit.materialId&0x80000000u)!=0u){return dryBodyPbrMaterialId(bodies[hit.materialId&0x7fffffffu]);}
  return hit.materialId;
}
fn dryPublishedMaterialValid(material:SvoMaterialRecord,index:u32)->bool{
  return dryMaterialPublished(material,index)&&(material.identity.w&SVO_MATERIAL_FLAG_OPAQUE)!=0u;
}
// The surface's material, from the published PBR record and nothing else.
//
// Two per-pixel procedural evaluations used to run here — a terrain colour
// policy and a general svoProceduralMaterial variation — both of them
// functions of the *world position* of the hit. That is the analytic dependency
// this cutover removes, in its purest form: a colour that cannot be answered
// from the voxel. What a surface looks like is now decided when the material
// record is authored, and read back through a 96-byte record by index.
fn dryEvaluateSurfaceMaterial(hit:DryHit,position:vec3f)->DrySurfaceMaterial {
  var materialId=dryResolvedMaterialId(hit);var baseOverride=vec3f(0.0);var useBaseOverride=false;var selectedEmission=vec3f(0.0);
  if((hit.materialId&0x80000000u)!=0u){let body=bodies[hit.materialId&0x7fffffffu];baseOverride=body.colorSelected.xyz;useBaseOverride=true;selectedEmission=body.colorSelected.w*vec3f(.12,.42,.32);}
  if(materialId>=dry.materialPublication.x){return dryInvalidSurfaceMaterial();}let material=dryMaterial(materialId);if(!dryPublishedMaterialValid(material,materialId)){return dryInvalidSurfaceMaterial();}
  var base=select(material.baseColorOpacity.xyz,baseOverride,useBaseOverride);var roughness=material.emissiveRoughness.w;var regionId=DRY_SURFACE_REGION_NONE;var variationFlags=0u;
  ${neutralSurfaceAlbedo ? /* wgsl */ `
  // Clay at 0.8, not 1.0. A unit-albedo surface clips to flat white under the
  // key light and loses precisely the shading this mode exists to show; 0.8 is
  // where plaster and porcelain actually sit, and it keeps a highlight's
  // rolloff on the curve instead of against the ceiling.
  //
  // The selection tint survives: it is a cursor affordance, not a material.
  // So does emission below, because an emitter is a light source and removing
  // it would change how the room is lit rather than what colour things are.
  base=select(vec3f(0.8),baseOverride,useBaseOverride);roughness=0.65;` : ""}
  return DrySurfaceMaterial(base,roughness,material.emissiveRoughness.xyz+selectedEmission,material.surface.x,vec3f(svoMaterialDielectricF0(material)),material.surface.y,regionId,variationFlags,1u,0u);
}
// The hover outline used to be applied here, and it is gone with the owner id
// it keyed off: a voxel no longer names the object it belongs to, so there is
// nothing for a cursor to select. uniforms.highlight is unread by this module.
fn shadeDryOpaque(hit:DryHit,ro:vec3f,rd:vec3f)->vec3f {
  if(hit.t>=DRY_MISS){return dryEnvironment(rd,0.0);}${screenSpaceProxyShadeWGSL}${prepassRadianceShortcutWGSL}${voxelLightCache ? "dryVoxelLightConsumerEligible=select(0u,1u,hit.motionKind==DRY_GBUFFER_MOTION_STATIC);" : ""}let position=ro+rd*hit.t;let surface=dryEvaluateSurfaceMaterial(hit,position);
  if(surface.valid==0u){return vec3f(0.0);}
  // What the pixel shades with and what a ray leaves along are two questions.
  // Every closure below keeps \`hit.normal\`; only the ray origins move to the
  // face the surface actually is, and only where a producer published one — for
  // every other hit this is \`hit.normal\`, the same float.
  let geometricNormal=dryGeometricNormal(hit);
  let directClosure=unifiedPbrMaterial(surface.baseColor,surface.metallic,surface.roughness,vec3f(0.0),0.0,surface.specularF0,surface.specularWeight,vec3f(0.0),0.0);var direct=vec3f(0.0);var sampleBudget=0u;
  let globalIllumination=(dry.materialPublication.w&${SVO_DRY_VISIBILITY_FLAGS.globalIllumination}u)!=0u;
  // GI supplements the authored lighting design; it must not replace every
  // local fixture with whichever record happens to be first (the garden's
  // first record is an intentionally weak dusk directional light). Keep all
  // configured emitters, while the sample-count selection below still limits
  // GLOBAL shading to one exact visibility sample per light.
  let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_LIGHT_MAXIMUM_RECORDS}u));
  for(var lightIndex=0u;lightIndex<${fastDeferred ? 1 : SVO_DRY_SCENE_MAX_SHADED_LIGHTS}u;lightIndex+=1u){
    if(lightIndex>=lightCount||sampleBudget>=dry.tuningCounts0.z){break;}${prepassLightSlotWGSL}${fastDeferred ? "var light=dryLighting.lights[lightIndex];light.identity.x=SVO_LIGHT_DIRECTIONAL;" : "let light=dryLighting.lights[lightIndex];"}if(light.identity.w!=dryLighting.metadata.y){continue;}let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA||light.identity.x==SVO_LIGHT_SPOT;let sampleCount=${fastDeferred ? "1u" : "select(select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area),1u,globalIllumination)"};
    for(var sampleIndex=0u;sampleIndex<${SVO_DRY_SCENE_AREA_LIGHT_SAMPLES}u;sampleIndex+=1u){if(sampleIndex>=sampleCount||sampleBudget>=dry.tuningCounts0.z){break;}sampleBudget+=1u;let sample=dryLightSample(light,sampleIndex,position);if(sample.valid==0u||dot(hit.normal,sample.towardLight)<=0.0){continue;}let visibility=dryLightVisibility(position,geometricNormal,hit.ownerId,sample.towardLight,sample.finiteDistance_m);let lighting=unifiedLightingInputWithGeometry(hit.normal,hit.normal,-rd,sample.towardLight,sample.radiance*visibility/f32(sampleCount));direct+=shadeUnifiedSurface(directClosure,lighting);}
  }
  let viewDirection=normalize(-rd);let reflected=reflect(rd,hit.normal);let diffuseColor=surface.baseColor*(1.0-surface.metallic);let f0=mix(surface.specularF0*surface.specularWeight,surface.baseColor,surface.metallic);let environmentBrdf=unifiedEnvironmentBrdf(max(dot(hit.normal,viewDirection),0.0),surface.roughness,f0);let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);let contactVisibility=dryContactVisibility(position,geometricNormal,hit.featureId,hit.ownerId);let ignoredBodyOwner=select(DRY_OWNER_NONE,hit.ownerId,hit.motionKind==DRY_GBUFFER_MOTION_RIGID);let gi=dryGlobalIllumination(position,hit.normal,ignoredBodyOwner);let diffuseVisibility=dryDiffuseMultiBounceVisibility(gi.visibility,diffuseColor);let diffuseEnvironmentScale=select(1.0,dry.giLighting.z,globalIllumination);let directScale=dry.giLighting.w;let diffuseEnvironment=diffuseColor*diffuseEnergy*svoEnvironmentDiffuseIrradiance(dryLighting.environment,hit.normal)*contactVisibility*diffuseVisibility*diffuseEnvironmentScale/UNIFIED_PI;let specularEnvironment=dryEnvironment(reflected,surface.roughness)*environmentBrdf;let indirectDiffuse=diffuseColor*gi.radiance;
  var shaded=max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*directScale+indirectDiffuse,vec3f(0.0));
  shaded*=dryVoxelFaceEdgeFactor(position,hit.normal,hit.t,hit.fieldSource);
  return shaded;
}
// Where the ray leaves the render voxel it just entered.
//
// This used to advance by the *largest* crossing a voxel can take — one whole
// cell along the ray's dominant axis — which is only the true exit for a ray
// that entered on that axis' face. Every other ray overshot, and an overshoot
// at a tank's base steps clean over the entry face of the floor the wall stands
// on and restarts the trace *inside* that solid, where the entered face is
// reconstructed from an interior point. The face it picks there is arbitrary,
// usually points away from the key light, and the wall's whole base shades
// black. Exiting exactly cannot skip a face, so the surface behind the glass is
// always entered where it is actually entered.
fn dryVoxelExit_m(position:vec3f,rd:vec3f)->f32{
  let cell=max(dry.mapping.cellSize,vec3f(1e-9));
  let nudge=1e-3*min(cell.x,min(cell.y,cell.z));
  // The hit sits *on* a voxel face, so name the voxel by a point just inside it
  // along the ray rather than by the face itself.
  let interior=position+rd*nudge;
  let base=dry.nodeMipOrigin.xyz+floor((interior-dry.nodeMipOrigin.xyz)/cell)*cell;
  let inverse=select(vec3f(DRY_MISS),vec3f(1.0)/rd,abs(rd)>vec3f(1e-9));
  let far=max((base-position)*inverse,(base+cell-position)*inverse);
  return max(nudge,min(far.x,min(far.y,far.z))+nudge);
}
fn dryTraceBeyondThinWall(hit:DryHit,ro:vec3f,rd:vec3f)->DryHit{
  var cursor=hit.t+dryVoxelExit_m(ro+rd*hit.t,rd);var behind=traceDrySolidSceneFrom(ro,rd,cursor);
  // A SolidWorld wall can span more than one render voxel. Treat the contiguous
  // run as one thin sheet, not as nested panes that multiply Fresnel eight times.
  for(var layer=0u;layer<16u;layer+=1u){
    if(!dryHitThinDielectric(behind)){break;}
    cursor=behind.t+dryVoxelExit_m(ro+rd*behind.t,rd);behind=traceDrySolidSceneFrom(ro,rd,cursor);
  }
  return behind;
}
fn shadeDryThinDielectric(hit:DryHit,ro:vec3f,rd:vec3f)->vec3f{
  var surface=hit;var throughput=vec3f(1.0);var color=vec3f(0.0);
  // A ray may cross both tank walls. Resolve a bounded sequence iteratively;
  // this is deliberately not recursive shader control flow.
  for(var sheet=0u;sheet<4u;sheet+=1u){
    // Invalid glass still behaves as a stopping surface. Once the material is
    // valid, the continuation below replaces this with the first opaque hit.
    drySurfaceOcclusionDepth_m=surface.t;
    let materialId=dryResolvedMaterialId(surface);if(materialId>=dry.materialPublication.x){return vec3f(0.0);}
    let material=dryMaterial(materialId);if(!dryMaterialThinDielectric(material,materialId)){return vec3f(0.0);}
    let f0=svoMaterialDielectricF0(material);let cosine=clamp(abs(dot(normalize(surface.normal),normalize(rd))),0.0,1.0);let fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
    color+=throughput*dryEnvironment(reflect(rd,surface.normal),material.emissiveRoughness.w)*fresnel;
    throughput*=dryThinDielectricTransmittance(material,surface.normal,rd);
    let behind=dryTraceBeyondThinWall(surface,ro,rd);
    if(!dryHitThinDielectric(behind)){
      // This is the depth at which the scene actually occludes water. A miss
      // publishes zero, the compact G-buffer's far-sentinel encoding.
      drySurfaceOcclusionDepth_m=select(0.0,behind.t,behind.t<DRY_MISS);
      return max(color+throughput*shadeDryOpaque(behind,ro,rd),vec3f(0.0));
    }
    surface=behind;
  }
  drySurfaceOcclusionDepth_m=0.0;
  return max(color+throughput*dryEnvironment(rd,0.0),vec3f(0.0));
}
fn shadeDrySurface(hit:DryHit,ro:vec3f,rd:vec3f)->vec3f{
  drySurfaceOcclusionDepth_m=select(0.0,hit.t,hit.t<DRY_MISS);
  ${fastDeferred ? "" : "if(dryHitThinDielectric(hit)){return shadeDryThinDielectric(hit,ro,rd);}"}
  return shadeDryOpaque(hit,ro,rd);
}
struct DryGlassSurface{color:vec3f,depth:f32,materialId:u32,ownerId:u32,paneId:u32,_padding:u32}
fn shadeThinGlass(glass:DryGlassHit,opaque:DryHit,ro:vec3f,rd:vec3f)->DryGlassSurface {
  let record=dryGlassPane(glass.recordIndex);let incidentIor=dryThinGlassIncidentIor();let optics=svoThinGlassOptics(record,glass.hit,incidentIor);
  // A collapsed sheet has no net Snell bend, so the already-resolved collinear
  // opaque hit is exactly the transmitted scene query; never traverse it twice.
  let reflected=dryEnvironment(reflect(rd,glass.hit.geometricNormal),.04);let transmitted=shadeDrySurface(opaque,ro,rd);
  let color=reflected*optics.fresnel+transmitted*optics.netTransmittance;
  return DryGlassSurface(color,glass.hit.t_m,svoThinGlassMaterialId(record),svoThinGlassOwnerId(record),svoThinGlassPaneId(record),0u);
}

fn dryThickGlassEmission(materialId:u32)->vec3f{
  if(materialId>=dry.materialPublication.x){return vec3f(0.0);}let material=dryMaterial(materialId);
  if(!svoMaterialValid(material,materialId)||material.identity.y!=dry.materialPublication.y){return vec3f(0.0);}return material.emissiveRoughness.xyz;
}
fn shadeThickGlass(glass:DryThickGlassHit,ro:vec3f,rd:vec3f)->DryGlassSurface{
  let record=thickGlass.records[glass.recordIndex];let first=dryThickGlassFirst(glass.interval);let ior=record.radiiYzIorEpsilon.z;
  let fromIor=select(1.0,ior,glass.interval.insideAtStart!=0u);let toIor=select(ior,1.0,glass.interval.insideAtStart!=0u);
  let firstOptics=svoThickGlassInterface(record,first,rd,fromIor,toIor,0.0);let reflected=dryEnvironment(firstOptics.reflectedDirection,.04);var transmitted=vec3f(0.0);var transmission=vec3f(0.0);
  if(firstOptics.totalInternalReflection==0u){
    if(glass.interval.insideAtStart!=0u){let origin=first.position_m+firstOptics.refractedDirection*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(origin,firstOptics.refractedDirection);transmitted=shadeDrySurface(opaque,origin,firstOptics.refractedDirection);transmission=vec3f(1.0-firstOptics.fresnel);}
    else if(glass.interval.tangent!=0u){let origin=first.position_m+rd*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(origin,rd);transmitted=shadeDrySurface(opaque,origin,rd);transmission=vec3f(1.0-firstOptics.fresnel);}
    else{
      let insideOrigin=first.position_m+firstOptics.refractedDirection*record.radiiYzIorEpsilon.w;let inside=svoThickGlassIntersect(record,insideOrigin,firstOptics.refractedDirection,0.0,record.absorptionPath.w,thickGlass.metadata.y);
      if(inside.status==SVO_THICK_GLASS_HIT){let exitSurface=inside.exit;let exitOptics=svoThickGlassInterface(record,exitSurface,firstOptics.refractedDirection,ior,1.0,inside.opticalPath_m);
        if(exitOptics.totalInternalReflection==0u){let outsideOrigin=exitSurface.position_m+exitOptics.refractedDirection*record.radiiYzIorEpsilon.w;let opaque=traceOpaqueScene(outsideOrigin,exitOptics.refractedDirection);transmitted=shadeDrySurface(opaque,outsideOrigin,exitOptics.refractedDirection);transmission=exitOptics.absorptionTint*(1.0-firstOptics.fresnel)*(1.0-exitOptics.fresnel);}
      }
    }
  }
  let materialId=svoThickGlassMaterialId(record);let color=reflected*firstOptics.fresnel+transmitted*transmission+dryThickGlassEmission(materialId);
  return DryGlassSurface(max(color,vec3f(0.0)),first.t_m,materialId,svoThickGlassOwnerId(record),svoThickGlassId(record),0u);
}

struct VertexOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}

struct DryFragmentOut{
  @location(0) radianceDepth:vec4f,
  @location(1) packedSurface:vec4u,
  @location(2) identityMedia:vec4u,
  @builtin(frag_depth) hardwareDepth:f32,
}
// Scene surfaces follow the complete renderer publication. Fluid and rigid
// paths publish their own local generations below.
fn dryPublicationGeneration()->u32{return dry.primitiveCandidates.w;}
struct DryRigidMotionSurface{velocity_m_s:vec3f,generation:u32,valid:u32}
fn dryRigidMotionSurface(hit:DryHit,worldSurfacePosition_m:vec3f)->DryRigidMotionSurface{
  if(hit.motionKind!=DRY_GBUFFER_MOTION_RIGID||hit.ownerId>=12u||hit.ownerId>=u32(round(uniforms.options.z))){return DryRigidMotionSurface(vec3f(0.0),dryPublicationGeneration(),0u);}
  let record=rigidMotion[hit.ownerId];let generation=svoPrimitiveMotionGeneration(record);let identityValid=record.identityRevision.x==hit.ownerId&&svoPrimitiveMotionOwnerId(record)==hit.ownerId&&svoPrimitiveMotionMaterialId(record)==dryResolvedMaterialId(hit)&&generation!=0u;let transformValid=distance(record.currentPositionDt.xyz,bodies[hit.ownerId].positionRadius.xyz)<=1e-5;let velocity=svoPrimitiveMotionVelocityAt(record,worldSurfacePosition_m);let valid=identityValid&&transformValid&&velocity.valid!=0u;
  return DryRigidMotionSurface(select(vec3f(0.0),velocity.velocity_m_s,valid),select(dryPublicationGeneration(),generation,generation!=0u),select(0u,1u,valid));
}
fn dryMediumPair(rd:vec3f,geometricNormal:vec3f,surfaceMedium:u32)->vec2u{
  return select(vec2u(surfaceMedium,DRY_MEDIUM_AIR),vec2u(DRY_MEDIUM_AIR,surfaceMedium),dot(rd,geometricNormal)<0.0);
}
fn dryHardwareDepth(t_m:f32,rd:vec3f,forward:vec3f)->f32{
  if(!(t_m<DRY_MISS)){return 0.0;}let viewDepth_m=t_m*max(dot(rd,forward),1e-6);return clamp(DRY_REVERSED_Z_NEAR_M/viewDepth_m,0.0,1.0);
}
fn dryFragmentOut(targets:SvoGBufferTargets,hardwareDepth:f32)->DryFragmentOut{
  return DryFragmentOut(targets.radianceDepth,targets.packedSurface,targets.identityMedia,hardwareDepth);
}

@fragment fn fragmentMain(input:VertexOut)->DryFragmentOut {
  let ndc=input.uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());dryVisibilityIgnoredBody=DRY_OWNER_NONE;dryThickGlassFailure=0u;
  // Curved thick glass is compiled separately from this Metal-sensitive pass.
  // Its authored pane therefore remains visible through the exact thin fallback.
  dryThickGlassEnabled=0u;
  let opaque=traceOpaqueScene(ro,rd);${prepassResolveCallWGSL}let glass=traceGlass(ro,rd,0.0,opaque.t);var color=shadeDrySurface(opaque,ro,rd);var depth=drySurfaceOcclusionDepth_m;
  let glassVisible=glass.hit.valid!=0u&&glass.hit.t_m<opaque.t;var glassSurface=DryGlassSurface(vec3f(0.0),DRY_MISS,0u,DRY_OWNER_NONE,0u,0u);
  if(glassVisible){glassSurface=shadeThinGlass(glass,opaque,ro,rd);color=glassSurface.color;depth=glassSurface.depth;}
  let vignette=1.0-.14*dot(ndc*.58,ndc*.58);let radiance=max(color*vignette,vec3f(0.0));let generation=dryPublicationGeneration();
  if(glassVisible){
    let media=dryMediumPair(rd,glass.hit.geometricNormal,DRY_MEDIUM_GLASS);
    let targets=svoGBufferSurface(radiance,depth,glass.hit.geometricNormal,glass.hit.geometricNormal,vec4u(glassSurface.materialId,glassSurface.ownerId,media.x,media.y),vec3f(0.0),DRY_GBUFFER_MOTION_STATIC,DRY_GBUFFER_FIELD_ANALYTIC,generation,SVO_GBUFFER_MOTION_VALID|svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_GLASS),SVO_FEATURE_SMOOTH);
    return dryFragmentOut(targets,dryHardwareDepth(depth,rd,forward));
  }
  if(opaque.t<DRY_MISS){
    let voxelGlass=dryHitThinDielectric(opaque);let media=dryMediumPair(rd,opaque.normal,select(DRY_MEDIUM_OPAQUE,DRY_MEDIUM_GLASS,voxelGlass));let rigidSurface=dryRigidMotionSurface(opaque,ro+rd*opaque.t);let motionVelocity=select(vec3f(0.0),rigidSurface.velocity_m_s,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionGeneration=select(generation,rigidSurface.generation,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let motionValid=select(opaque.motionValid,rigidSurface.valid,opaque.motionKind==DRY_GBUFFER_MOTION_RIGID);let producer=select(SVO_GBUFFER_PRODUCER_TRACED,SVO_GBUFFER_PRODUCER_GLASS,voxelGlass);var flags=select(0u,SVO_GBUFFER_MOTION_VALID,motionValid!=0u)|svoGBufferProducerFlags(producer);if(opaque.featureId!=SVO_FEATURE_SMOOTH){flags|=DRY_GBUFFER_HARD_FEATURE;}
    let targets=svoGBufferSurface(radiance,depth,opaque.normal,opaque.normal,vec4u(dryResolvedMaterialId(opaque),opaque.ownerId,media.x,media.y),motionVelocity,opaque.motionKind,opaque.fieldSource,motionGeneration,flags,opaque.featureId);
    return dryFragmentOut(targets,dryHardwareDepth(opaque.t,rd,forward));
  }
  return dryFragmentOut(svoGBufferMiss(radiance,0u,generation,DRY_GBUFFER_NO_INTERSECTION,svoGBufferProducerFlags(SVO_GBUFFER_PRODUCER_TRACED)),0.0);
}
${splitEntryWGSL}${rasterPrimaryEntryWGSL}${rasterPrimary && experiments.surfaceMesh ? svoSurfaceMeshWGSL(splitGroup, SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals, experiments.surfaceMeshCulling !== false) : ""}${prepassEntryWGSL}${prepassFromPrimaryEntryWGSL}${pixelProbe ? createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions(
    traversalMode === "raster-primary" ? "raster" : "traced",
    {
      brickOccupancyMode,
      brickContour: brickContourPrimary && !brickContourInertProbe,
      brickContourEntryClamp,
      brickContourExitScope,
    },
  )) : ""}`;
  if (experiments.dropGiPageCache) {
    shader = shader.replace("var<private> dryGiPageCache:DryNodeMipPageCache;", "");
    const loadStart = shader.indexOf("fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{");
    const loadEnd = shader.indexOf("struct DryGlobalIllumination", loadStart);
    if (loadStart < 0 || loadEnd < 0) throw new Error("GI page-cache source drifted");
    const prefix = shader.slice(0, loadStart);
    const load = shader.slice(loadStart, loadEnd)
      .replace("{\n  if(!dryNodeMipReady())", "{\n  var pageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);\n  if(!dryNodeMipReady())")
      .replaceAll("dryGiPageCache", "pageCache");
    const suffix = shader.slice(loadEnd)
      .replaceAll("dryGiPageCache=DryNodeMipPageCache(vec3u(0u),0xffffffffu,vec3u(0u),0u,0u,0xffffffffu,0u);", "");
    shader = prefix + load + suffix;
  }
  if (experiments.halfPrecisionLighting) {
    const original = "var accumulated0=vec4f(0.0);var accumulated1=vec4f(0.0);var accumulated2=vec4f(0.0);var weightSum=0.0;"
      + "\n  var accumulatedRadiance=vec4f(0.0);var radianceWeightSum=0.0;var accumulatedGi=vec4f(0.0);var giWeightSum=0.0;";
    const half = "var accumulated0=vec4h(0.0h);var accumulated1=vec4h(0.0h);var accumulated2=vec4h(0.0h);var weightSum=0.0h;"
      + "\n  var accumulatedRadiance=vec4h(0.0h);var radianceWeightSum=0.0h;var accumulatedGi=vec4h(0.0h);var giWeightSum=0.0h;";
    if (!shader.includes(original)) throw new Error("Half-precision accumulator source drifted");
    shader = `enable f16;\n${shader.replace(original, half)
      .replace("accumulated0+=dryPrepassUnpack0(packed)*weight;", "accumulated0+=vec4h(dryPrepassUnpack0(packed))*f16(weight);")
      .replace("accumulated1+=dryPrepassUnpack1(packed)*weight;", "accumulated1+=vec4h(dryPrepassUnpack1(packed))*f16(weight);")
      .replace("accumulated2+=dryPrepassUnpack2(packed)*weight;", "accumulated2+=vec4h(dryPrepassUnpack2(packed))*f16(weight);")
      .replace("accumulatedGi+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;giWeightSum+=weight;",
        "accumulatedGi+=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0))*f16(weight);giWeightSum+=f16(weight);")
      .replace("accumulatedRadiance+=textureLoad(dryPrepassRadianceTexture,texel,0)*weight;radianceWeightSum+=weight;",
        "accumulatedRadiance+=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0))*f16(weight);radianceWeightSum+=f16(weight);")
      .replace("weightSum+=weight;", "weightSum+=f16(weight);")
      .replace("if(weightSum<", "if(f32(weightSum)<")
      .replace("dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;",
        "dryPrepassData0=accumulated0/weightSum;dryPrepassData1=accumulated1/weightSum;dryPrepassData2=accumulated2/weightSum;")
      .replace("if(giWeightSum>=", "if(f32(giWeightSum)>=")
      .replace("dryPrepassGi=accumulatedGi/giWeightSum;", "dryPrepassGi=accumulatedGi/giWeightSum;")
      .replace("&&radianceWeightSum>1e-6", "&&f32(radianceWeightSum)>1e-6")
      .replace("dryPrepassRadiance=accumulatedRadiance/radianceWeightSum;",
        "dryPrepassRadiance=accumulatedRadiance/radianceWeightSum;")
      .replaceAll("var<private> dryPrepassData0:vec4f;", "var<private> dryPrepassData0:vec4h;")
      .replaceAll("var<private> dryPrepassData1:vec4f;", "var<private> dryPrepassData1:vec4h;")
      .replaceAll("var<private> dryPrepassData2:vec4f;", "var<private> dryPrepassData2:vec4h;")
      .replaceAll("var<private> dryPrepassRadiance:vec4f;", "var<private> dryPrepassRadiance:vec4h;")
      .replaceAll("var<private> dryPrepassGi:vec4f;", "var<private> dryPrepassGi:vec4h;")
      .replace("return dryPrepassData0[index];", "return f32(dryPrepassData0[index]);")
      .replace("return dryPrepassData1[index-4u];", "return f32(dryPrepassData1[index-4u]);")
      .replace("return dryPrepassData2[min(index-8u,3u)];", "return f32(dryPrepassData2[min(index-8u,3u)]);")
      .replaceAll("dryPrepassData0=vec4f(1.0);dryPrepassData1=vec4f(1.0);dryPrepassData2=vec4f(1.0);dryPrepassRadiance=vec4f(0.0);dryPrepassGi=vec4f(0.0,0.0,0.0,1.0);",
        "dryPrepassData0=vec4h(1.0h);dryPrepassData1=vec4h(1.0h);dryPrepassData2=vec4h(1.0h);dryPrepassRadiance=vec4h(0.0h);dryPrepassGi=vec4h(0.0h,0.0h,0.0h,1.0h);")
      .replace("dryPrepassData0.x*(prepassUnblocked/f32(prepassSamples))",
        "f32(dryPrepassData0.x)*(prepassUnblocked/f32(prepassSamples))")
      .replace("return max(dryPrepassRadiance.rgb,vec3f(0.0));",
        "return max(vec3f(dryPrepassRadiance.rgb),vec3f(0.0));")
      .replace("return DryGlobalIllumination(max(dryPrepassGi.rgb,vec3f(0.0)),clamp(dryPrepassGi.a,0.0,1.0),1u);",
        "return DryGlobalIllumination(max(vec3f(dryPrepassGi.rgb),vec3f(0.0)),clamp(f32(dryPrepassGi.a),0.0,1.0),1u);")
      .replaceAll("dryPrepassData0=dryPrepassUnpack0(packed);dryPrepassData1=dryPrepassUnpack1(packed);dryPrepassData2=dryPrepassUnpack2(packed);",
        "dryPrepassData0=vec4h(dryPrepassUnpack0(packed));dryPrepassData1=vec4h(dryPrepassUnpack1(packed));dryPrepassData2=vec4h(dryPrepassUnpack2(packed));")
      .replace("dryPrepassGi=textureLoad(dryPrepassRadianceTexture,texel,0);",
        "dryPrepassGi=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0));")
      .replace("dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,texel,0);",
        "dryPrepassRadiance=vec4h(textureLoad(dryPrepassRadianceTexture,texel,0));")
      .replace("dryPrepassRadiance=textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0);",
        "dryPrepassRadiance=vec4h(textureSampleLevel(dryPrepassRadianceTexture,nodeMipSampler,pixel/max(uniforms.viewport.xy,vec2f(1.0)),0.0));")
      .replace("dryPrepassRadiance=textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0);",
        "dryPrepassRadiance=vec4h(textureLoad(dryPrepassRadianceTexture,bestRadianceTexel,0));")}`;
  }
  return shader;
}


export const drySceneShader = createSvoDrySceneFragmentWGSL(1);


export const drySceneVertexShader = /* wgsl */ `
struct VertexOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOut {
  var points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var output:VertexOut;output.position=vec4f(points[index],0,1);output.uv=points[index]*.5+.5;return output;
}
`;


/**
 * Coverage-scaled thin-glass discovery for split rendering. Quads are only
 * conservative raster candidates: the fragment stage repeats the canonical
 * analytic finite-pane intersection before writing exact depth and identity.
 */
export const svoDryRasterGlassShader = /* wgsl */ `
${svoThinGlassWGSL}
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, reservedSceneMeta:vec4f, reservedSceneLanes:array<vec4f,16> }
struct GlassRasterParams { paneCount:u32, _padding0:u32, _padding1:u32, _padding2:u32 }
struct GlassRasterVertexOut {
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) recordIndex:u32,
}
struct GlassRasterFragmentOut {
  @location(0) glassKey:u32,
  @builtin(frag_depth) hardwareDepth:f32,
}
@group(0) @binding(0) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(4) var<storage,read> dryRasterSceneArena:array<u32>;
@group(1) @binding(0) var dryRasterOpaqueGeometry:texture_2d<f32>;
@group(1) @binding(1) var<uniform> glassRaster:GlassRasterParams;
const DRY_RASTER_GLASS_NEAR_M:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
const DRY_RASTER_GLASS_WORD_OFFSET:u32=${SVO_DRY_SCENE_ARENA_LAYOUT.glassOffsetBytes / 4}u;
fn dryRasterGlassWords4(offset:u32)->vec4u{return vec4u(dryRasterSceneArena[offset],dryRasterSceneArena[offset+1u],dryRasterSceneArena[offset+2u],dryRasterSceneArena[offset+3u]);}
fn dryRasterGlassPane(index:u32)->SvoThinGlassRecord{let base=DRY_RASTER_GLASS_WORD_OFFSET+index*${SVO_THIN_GLASS_RECORD_WORDS}u;return SvoThinGlassRecord(
  bitcast<vec4f>(dryRasterGlassWords4(base)),bitcast<vec4f>(dryRasterGlassWords4(base+4u)),bitcast<vec4f>(dryRasterGlassWords4(base+8u)),
  bitcast<vec4f>(dryRasterGlassWords4(base+12u)),dryRasterGlassWords4(base+16u));}
fn dryRasterGlassCorner(index:u32)->vec2f{
  var corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  return corners[index];
}
fn dryRasterGlassRay(pixel:vec2f)->mat2x3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());return mat2x3f(ro,rd);
}
@vertex fn glassRasterVertex(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) recordIndex:u32)->GlassRasterVertexOut{
  let record=dryRasterGlassPane(recordIndex);let corner=dryRasterGlassCorner(vertexIndex);let padding=max(2.0*record.extentIorEpsilon.w,1e-5);let local=vec3f(corner*(record.extentIorEpsilon.xy+vec2f(padding)),0.0);let world=record.centerThickness.xyz+svoThinGlassRotate(record.orientation,local);
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));let relative=world-ro;let viewDepth=dot(relative,forward);let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  return GlassRasterVertexOut(vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),.5*viewDepth,viewDepth),recordIndex);
}
@fragment fn glassRasterFragment(input:GlassRasterVertexOut)->GlassRasterFragmentOut{
  if(input.recordIndex>=glassRaster.paneCount){discard;}
  let record=dryRasterGlassPane(input.recordIndex);
  let coordinate=vec2i(input.position.xy);let opaqueDepth=textureLoad(dryRasterOpaqueGeometry,coordinate,0).w;let ray=dryRasterGlassRay(input.position.xy);let hit=svoThinGlassIntersect(record,ray[0],ray[1],0.0,opaqueDepth,1e-6,record.extentIorEpsilon.w);if(hit.valid==0u||!(hit.t_m<opaqueDepth)){discard;}
  let forward=normalize(uniforms.cameraTarget.xyz-uniforms.cameraPosition.xyz);let viewDepth=hit.t_m*max(dot(ray[1],forward),1e-6);let hardwareDepth=clamp(DRY_RASTER_GLASS_NEAR_M/viewDepth,0.0,1.0);return GlassRasterFragmentOut(input.recordIndex+1u,hardwareDepth);
}
`;


/**
 * Slack on the scene-primitive march bracket, absolute and relative to the span.
 *
 * Exported so the CPU containment test brackets with exactly the shipped
 * numbers rather than a tolerance of its own, which is the only way that test
 * can fail for a real reason.
 */
export const SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_M = 1e-4;

export const SVO_SCENE_PRIMITIVE_MARCH_SPAN_PAD_RELATIVE = 1e-3;


export const SVO_SCENE_PRIMITIVE_RASTER_CONTRACT = Object.freeze({
  verticesPerProxy: 36,
  /**
   * One face of the proxy box, not both.
   *
   * The fragment repeats the shared exact intersection, so the two faces of a
   * box produce the same hit and the second one is pure duplicated work — and
   * this is the pass every marched primitive lands in, where that work is a
   * sphere trace rather than a quadratic. Drawing the far side (and not the
   * near) is what keeps a camera *inside* a primitive's proxy shading it, which
   * is the same reason the brick raster culls this way: the corner table winds
   * outward-CCW in world space and projection flips the near-facing triangles
   * to clockwise, which WebGPU's default `ccw` front face calls back faces.
   *
   * The near-plane case is unaffected: a proxy that contains the eye emits a
   * full-screen triangle instead, and that one is wound front-facing.
   */
  cullMode: SVO_BRICK_RASTER_CONTRACT.cullMode,
  /**
   * Fixed per-pixel conservative candidate arena for the authored SDF set.
   *
   * WGSL has no conservative depth (gpuweb#5342), so the direct fragment — which
   * writes `frag_depth` *and* discards — lets the tiler reject nothing and every
   * covering proxy marches. The hero framing puts a median of 7 proxies over a
   * covered pixel, p99 33 and max 59; forty entries keep the overflow arm below
   * one percent of covered pixels while capacity stays a performance parameter
   * and never an image change. Deliberately larger than the brick arena's 24 and
   * *not* built on its 32-entry u32 `visited` mask, which p99=33 already exceeds:
   * candidates here carry their own ordering key and are walked by monotone
   * extraction instead.
   */
  coverageCandidatesPerPixel: 40,
  /**
   * Candidate key = floored 16-bit entry distance over a 16-bit record index.
   *
   * Truncating the low half of a non-negative float's bit pattern is monotone
   * and rounds *down*, so ordering by the packed word orders by a conservative
   * lower bound on the ray's entry into the primitive's oriented box. That is
   * exactly what front-to-back early termination needs: a candidate whose lower
   * bound is already behind the best hit cannot produce a nearer one. Sixteen
   * bits of index cap the arena at 65,536 authored records; past that the
   * encode falls back to the historical direct pass rather than aliasing.
   */
  coverageIndexBits: 16,
  entryPoints: Object.freeze({
    vertex: "dryScenePrimitiveRasterVertex",
    /** The same proxy, behind the near-field band's membership bit. */
    bandVertex: "dryScenePrimitiveBandVertex",
    fragment: "dryScenePrimitiveRasterFragment",
    coverage: "dryScenePrimitiveCoverageFragment",
    resolve: "dryScenePrimitiveCoverageResolveFragment",
    overflowResolve: "dryScenePrimitiveCoverageOverflowFragment",
  }),
} as const);


/** Records addressable by a coverage candidate key; beyond it the direct pass runs. */
export const SVO_SCENE_PRIMITIVE_COVERAGE_MAXIMUM_RECORDS =
  1 << SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageIndexBits;


export const svoDrySceneShader = drySceneShader;

export const svoDrySceneVertexShader = drySceneVertexShader;
