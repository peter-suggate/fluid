import {sourceMovingSolidBodies} from '../source-moving-solids';
import { defaultScene, type SceneDescription, type Vec3 } from '../../core/model';
import { defineScene, type SceneDefinition } from '../../core/scene-definition';
import { cachedClothTopology, loadClothTopology } from '../source-assets';
import { sourceFieldBounds } from '../source-motion';
import { sourceInitialLiquid } from './initial-liquid';
import { SourceEmissionScheduler } from '../source-fields';
import { withStaticSourceSolids } from '../source-static-solids';
import { loadSourceMesh, SPOT_SOURCE_ASSET } from '../source-mesh';
import type { ClothVector } from '../types';
import type { WetClothSourceDefinition } from '../source-types';
import { sourceClothDocument, wetClothSource, wetClothWaterSources,wetClothVariantSources } from './sources';
import {wetClothVariantTopology} from './variants';

/** Page pools authored from a measured peak (no runtime growth). The free-space
 * default (100 blocks) is exhausted at commit 1 by pore-test-mid's falling
 * ball: measured with 200 blocks, commit 1 held 105 (72 resident including 45
 * claimed, plus 33 retired seed pages in their hold); samples every 50 steps
 * through impact (0.12 s) peaked at 84 resident. 128 = peak + 22% margin for
 * transients between those samples. */
const WET_CLOTH_PAGE_POOL_BLOCKS: Readonly<Record<string, number>> = {'wetcloth-source-pore-test-mid': 128};

const vec = (v: ClothVector): Vec3 => ({x: v[0], y: v[1], z: v[2]});
const plus = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** Admission requirements, consumed by runtime gates. Importing a reference
 * input never enables a solver capability or certifies a paper result. */
export function wetClothSourceCapabilities(source: WetClothSourceDefinition): readonly string[] {
  const features = new Set(['free-space-boundary']);
  if (source.counts.triangles) features.add('cloth-shell-mechanics');
  if (source.counts.rods) features.add('cloth-rod-mechanics');
  if(source.counts.rods&&source.materials.some(m=>m.radius_m!==m.radiusB_m))features.add('elliptical-yarn-geometry');
  if (source.scripts.length) features.add('source-cubic-motion');
  if (source.fields.some(f => f.usage === 'solid')) features.add('source-solid-collision');
  if (source.fields.some(f => f.usage === 'terminator')) features.add('source-termination');
  if (source.fields.some(f => f.usage === 'source')) {
    features.add('narrowband-flip');
    if (sourceInitialLiquid(source.fields, source.scripts).needsSourceEmitter) features.add('source-sdf-emission');
    if (source.counts.vertices) { features.add('cloth-liquid-two-way'); features.add('cloth-porous-transport'); }
  }
  if (source.fields.some(f => f.kind === 'file')) features.add('source-external-mesh');
  if (source.fields.some(f => f.kind === 'intersect')) features.add('source-csg');
  return [...features];
}

/** Caller must load the source asset before synchronous sceneDocument/build.
 * This is deliberately separate from the main runnable scene catalog. */
export async function prepareWetClothSource(id: string, request?: typeof fetch): Promise<void> {
  const source = wetClothSource(id);
  await Promise.all([loadClothTopology(source.asset, request), ...source.fields.filter(f=>f.kind==='file').map(f=>loadSourceMesh(f.filename!,request))]);
}

export function buildWetClothSourceScene(id: string): SceneDescription {
  const source = wetClothSource(id),original = cachedClothTopology(source.asset);
  if (!original) throw new Error(`Load ${id} with prepareWetClothSource before building its scene`);
  const data=wetClothVariantTopology(original,source.variant);
  const missing = source.externalDependencies.filter(d => !d.available_in_release && d.path !== SPOT_SOURCE_ASSET.filename);
  if (missing.length) throw new Error(`Missing exact source geometry: ${missing.map(d => d.path).join(', ')}`);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const include = (p: ClothVector) => { for (let i = 0; i < 3; i++) { min[i] = Math.min(min[i]!, p[i]!); max[i] = Math.max(max[i]!, p[i]!); } };
  for (let i = 0; i < data.restPositions_m.length; i += 3) include(data.restPositions_m.slice(i, i + 3) as unknown as ClothVector);
  for (const f of source.fields.filter(f => f.root && f.usage !== 'terminator')) {
    const bounds = sourceFieldBounds(source.fields, f.index); include(bounds.min); include(bounds.max);
  }
  if (!Number.isFinite(min[0]!)) throw new Error(`Source ${id} has no finite geometry`);
  const margin = 8 * source.cellSize_m;
  const offset: ClothVector = [(-(max[0]! + min[0]!) / 2) || 0, margin - min[1]!, (-(max[2]! + min[2]!) / 2) || 0];
  const dimensions = max.map((v, i) => Math.ceil((v - min[i]! + 2 * margin) / (32 * source.cellSize_m)) * 32 * source.cellSize_m);
  const scene = structuredClone(defaultScene);
  scene.sceneId = id; scene.duration_s = source.duration_s;
  scene.systems = {fluid: source.fields.some(f => f.usage === 'source')};
  scene.container = {width_m: dimensions[0]!, height_m: dimensions[1]!, depth_m: dimensions[2]!, fillFraction: 0,
    top: 'open', boundary: 'free-space', fluidWallMode: 'free-slip', vessel: 'none'};
  // The plugin owns physical boundaries. This finite envelope is allocation
  // metadata; free-space-boundary must be admitted before any simulation.
  scene.voxelDomain = {finestCellSize_m: source.cellSize_m, brickSize_cells: 8,
    bounds_m: {min: {x: -dimensions[0]! / 2, y: 0, z: -dimensions[2]! / 2}, max: {x: dimensions[0]! / 2, y: dimensions[1]!, z: dimensions[2]! / 2}}};
  scene.solidVoxels = []; scene.rigidBodies = [];
  scene.scenery = {palettes: {}, nodes: [{id: 'source-empty-enclosure', kind: 'room-shell', materialModel: 'room', faces: [], floor: {colorLinear: [1,1,1]}, wall: {colorLinear: [1,1,1]}, ceiling: {colorLinear: [1,1,1]}}]}; delete scene.terrain; delete scene.solidBoundary;
  scene.fluid = {density_kg_m3: source.liquid.density_kg_m3,
    dynamicViscosity_Pa_s:Number(source.sourceSettings.liquid.computeViscosity??0)!==0?source.liquid.dynamicViscosity_Pa_s:0,
    // The source's free-surface force is independently disabled by default.
    // Capillary/cohesive material gamma remains in sourceClothDocument.
    surfaceTension_N_m: Number(source.sourceSettings.liquid.surfTension??0)!==0?source.liquid.surfaceTension_N_m:0, gravity_m_s2: vec(source.gravity_m_s2),
    initialCondition: 'tank-fill'};
  scene.numerics = {...scene.numerics, fixedDt_s: source.dt_s, maxDt_s: source.dt_s};
  scene.nominalResolution = {length_m: source.cellSize_m};
  if (source.counts.vertices) {
    const cloth = sourceClothDocument(id);
    scene.cloth = [{...cloth, geometry: {kind: 'inline', data: {...data,
      restPositions_m: data.restPositions_m.map((v, i) => v + offset[i % 3]!) }},
      scripts: cloth.scripts?.map(s => ({...s, ...(s.origin_m ? {origin_m: plus(s.origin_m, offset)} : {})}))}];
  } else scene.cloth = [];
  scene.wetClothSource = {schemaVersion: 1, sourceId: source.id, sourceOffset_m: offset, timeStep_s: source.dt_s,
    ...(source.variant?{variant:source.variant}:{}),
    requiredCapabilities: wetClothSourceCapabilities(source),
    fields: source.fields.map(f => ({...f, ...(f.center_m ? {center_m: plus(f.center_m, offset)} : {})})),
    scripts: source.scripts.map(s => ({...s, ...(s.origin_m ? {origin_m: plus(s.origin_m, offset)} : {})}))};
  const initial = sourceInitialLiquid(scene.wetClothSource.fields, scene.wetClothSource.scripts);
  const poolBlocks = WET_CLOTH_PAGE_POOL_BLOCKS[source.id];
  if (poolBlocks) scene.fluid.pagePool = {blocks: poolBlocks};
  if (initial.volumes.length) {
    scene.fluid.initialLiquidVolumes = initial.volumes;
    scene.fluid.initialVelocity_m_s = initial.velocity_m_s;
  }
  if (initial.scheduledDrops.length) scene.fluid.scheduledDrops = initial.scheduledDrops;
  scene.wetClothSource = {...scene.wetClothSource, initializedFieldIndices: initial.initializedFieldIndices, scheduledFieldIndices: initial.scheduledFieldIndices};
  const allEvents = new SourceEmissionScheduler(scene.wetClothSource.fields).advance(0, source.duration_s + source.dt_s);
  const atZero = allEvents.filter(e => e.start_s === 0);
  const velocities = atZero.map(e => e.velocity_m_s), velocity = velocities[0] ?? [0,0,0];
  const compatible = velocities.every(v => v.every((x,i) => Math.abs(x-velocity[i]!)<1e-12));
  const initialSdfFieldIndices = compatible ? [...new Set(atZero.map(e=>e.fieldIndex))].filter(i=>!initial.initializedFieldIndices.includes(i)) : [];
  if (initialSdfFieldIndices.length) scene.fluid.initialVelocity_m_s = {x:velocity[0]!,y:velocity[1]!,z:velocity[2]!};
  const initialDone = new Set([...initial.initializedFieldIndices,...initialSdfFieldIndices]);
  const pending = allEvents.some(e => e.kind === 'interval' || (e.start_s === 0 ? !initialDone.has(e.fieldIndex)
    : !initial.scheduledFieldIndices.includes(e.fieldIndex)));
  const requiredCapabilities = [...scene.wetClothSource.requiredCapabilities.filter(c=>c!=='source-sdf-emission')];
  if (pending) requiredCapabilities.push('source-sdf-emission');
  if (initialSdfFieldIndices.length) requiredCapabilities.push('source-initial-sdf');
  scene.wetClothSource = {...scene.wetClothSource, initialSdfFieldIndices, requiredCapabilities};
  const built=withStaticSourceSolids(scene),moving=sourceMovingSolidBodies(built.wetClothSource!,scene.cloth?.[0]?.materials[0]?.friction??0);
  if(!moving.length)return built;
  const groups=new Set(built.wetClothSource!.scripts.map(s=>s.group));
  return {...built,rigidBodies:[...built.rigidBodies,...moving],wetClothSource:{...built.wetClothSource!,movingSolidFieldIndices:built.wetClothSource!.fields.filter(f=>f.root&&f.usage==='solid'&&groups.has(f.group)).map(f=>f.index)}};
}

/** All released water inputs have definitions, but these are reference fixtures
 * until the required physics and shared-geometry gates have passed. */
export const wetClothSourceDefinitions: readonly SceneDefinition[] = wetClothWaterSources.map(source => defineScene({
  id: source.id, name: `Source fixture: ${source.name}`, blurb: 'Imported libWetCloth input; simulation reproduction has not been validated.',
  audience: 'validation', shelf: 'Liquid–fabric source fixtures', environment: 'default', containerShell: 'authored',
  methodProfile: {methodId: 'uniform-narrow-band-flip', quality: 'high', overrides: {timeStep: 'scene',...(source.fields.some(f=>f.usage==='terminator')?{bulkTier:'off'}:{})}},
  build: () => buildWetClothSourceScene(source.id),
}));

export const wetClothVariantDefinitions:readonly SceneDefinition[]=wetClothVariantSources.map(source=>defineScene({
 id:source.id,name:`${source.variant!.kind==='exploratory'?'Exploratory':'Paper-corrected'} fixture: ${source.name}`,
 blurb:`${source.variant!.reason} Simulation reproduction has not been validated.`,audience:'validation',shelf:'Liquid–fabric parameter variants',environment:'default',containerShell:'authored',
 methodProfile:{methodId:'uniform-narrow-band-flip',quality:'high',overrides:{timeStep:'scene',...(source.fields.some(f=>f.usage==='terminator')?{bulkTier:'off'}:{})}},
 build:()=>buildWetClothSourceScene(source.id),
}));
export const wetClothValidationDefinitions:readonly SceneDefinition[]=[...wetClothSourceDefinitions,...wetClothVariantDefinitions];
