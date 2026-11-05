import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { cachedClothTopology, decodeClothTopology, loadClothTopology } from './source-assets';
import { sourceClothDocument, wetClothSource, wetClothWaterSources } from './scenes/sources';
import { sourceFieldDistance, sourceTerminatesPoint, SourceEmissionScheduler } from './source-fields';
import type { WetClothSourceField } from './source-types';

const readAsset = async (url: string) => new Uint8Array(await readFile(`public${url}`));
const field = (overrides: Partial<WetClothSourceField> = {}): WetClothSourceField => ({index: 0, kind: 'sphere', usage: 'source', group: 0, inside: false, sampled: true, root: true, center_m: [0, 0, 0], radius_m: 1, sourceAttributes: {}, ...overrides});

test('source viscosity switches and extensional/shear factors map to physical damping modes',()=>{
 const towel=sourceClothDocument('wetcloth-source-tighten-towel').materials[0]!,ratio=2780/1830000;
 assert.equal(towel.rodRadius_m,.000165);assert.ok(Math.abs(towel.rodRadiusB_m!-.00018)<1e-18);assert.ok(Math.abs(towel.thickness_m-.000345)<1e-18);
 assert.equal(towel.membraneDamping_s,3*ratio);assert.equal(towel.bendingDamping_s,3*ratio);
 assert.equal(towel.rodStretchDamping_s,3*ratio);assert.equal(towel.rodBendingDamping_s,3*ratio);
 assert.equal(towel.rodTwistDamping_s,2*(1+.202)*ratio);
 const bendingOnly=sourceClothDocument('wetcloth-source-buoyancy-test-with').materials[0]!;
 assert.equal(bendingOnly.membraneDamping_s,0);assert.equal(bendingOnly.rodStretchDamping_s,0);
 assert.ok(bendingOnly.bendingDamping_s!>0);assert.ok(bendingOnly.rodTwistDamping_s!>0);
 const disabled=sourceClothDocument('wetcloth-source-simple-yarn-loop').materials[0]!;
 for(const key of ['membraneDamping_s','bendingDamping_s','rodStretchDamping_s','rodBendingDamping_s','rodTwistDamping_s'] as const)assert.equal(disabled[key],0);
});

test('source free-surface tension switch is independent from capillary material gamma',async()=>{
 const {prepareWetClothSource,buildWetClothSourceScene}=await import('./scenes/definitions');
 const request=(async(url:string)=>new Response(await readAsset(url))) as typeof fetch;
 for(const id of ['wetcloth-source-pore-test-mid','wetcloth-source-double-surf-tension','wetcloth-source-simple-surf-tension','wetcloth-paper-figure10j-90deg']){
  await prepareWetClothSource(id,request);const scene=buildWetClothSourceScene(id),source=wetClothSource(id);
  const enabled=Number(source.sourceSettings.liquid.surfTension??0)!==0;
  assert.equal(scene.fluid.surfaceTension_N_m,enabled?source.liquid.surfaceTension_N_m:0,id);
  assert.equal(scene.fluid.dynamicViscosity_Pa_s,0,'released water fixtures disable bulk viscosity');
  for(const material of scene.cloth?.flatMap(d=>d.materials)??[]){assert.equal(material.surfaceTension_N_m,source.liquid.surfaceTension_N_m);assert.equal(material.waterViscosity_Pa_s,source.liquid.dynamicViscosity_Pa_s);}
 }
 assert.equal(sourceClothDocument('wetcloth-paper-figure10j-90deg').materials[0]!.surfaceTension_N_m,.072);
});

test('all 36 water source assets verify exact counts, SHA and topology', async () => {
  assert.equal(wetClothWaterSources.length, 36);
  assert.ok(wetClothWaterSources.every(s => !/honey|acetal|pore-test-oil/.test(s.id)));
  for (const s of wetClothWaterSources) {
    const bytes = gunzipSync(await readAsset(s.asset.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), s.asset.sha256, s.id);
    const data = decodeClothTopology(bytes);
    assert.equal(data.restPositions_m.length, s.counts.vertices * 3, s.id);
    assert.equal(data.triangles.length, s.counts.triangles * 3, s.id);
    assert.equal(data.rodOffsets.length - 1, s.counts.rods, s.id);
    if (s.counts.vertices) assert.equal(sourceClothDocument(s.id).geometry, s.asset);
  }
});

test('asset loader shares verified source topology and rejects corrupt references', async () => {
  const s = wetClothWaterSources.find(s => s.id.endsWith('simple-cloth'))!;
  let requests = 0;
  const request = (async () => { requests++; return new Response(await readAsset(s.asset.url)); }) as typeof fetch;
  const [a, b] = await Promise.all([loadClothTopology(s.asset, request), loadClothTopology(s.asset, request)]);
  assert.equal(a, b); assert.equal(cachedClothTopology(s.asset), a); assert.equal(requests, 1);
  await assert.rejects(loadClothTopology({...s.asset, vertexCount: 0}, request), /vertex count/);
  await assert.rejects(loadClothTopology({...s.asset, sha256: '0'.repeat(64)}, request), /checksum/);
  assert.throws(() => decodeClothTopology(new Uint8Array(20)), /Unknown/);
  assert.throws(() => wetClothSource('honey'), /Unknown/);
});

test('source masks and nonuniform solid fractions are retained', async () => {
  const decode = async (suffix: string) => decodeClothTopology(gunzipSync(await readAsset(wetClothWaterSources.find(s => s.id.endsWith(suffix))!.asset.url)));
  const loop = await decode('simple-yarn-loop');
  assert.ok(loop.fixed!.includes(2)); assert.ok(loop.fixed!.includes(3));
  const ring = await decode('ring-test');
  assert.ok(ring.solidFraction!.every(v => v === 1));
  const nonuniform = await decode('ring-test-nonuniform');
  assert.ok(new Set(nonuniform.solidFraction).size > 1);
  const fuzzy = await decode('fuzzy-cloth');
  const shell = new Set(fuzzy.triangles);
  assert.ok(fuzzy.rodOffsets.slice(0, -1).some(offset => shell.has(fuzzy.rodIndices[offset]!)), 'pile roots share shell addresses');
});

test('source signed distance matches rounded/rotated shapes, intersection and terminator signs', () => {
  const box = field({kind: 'box', halfExtents_m: [1, 2, 3], roundingRadius_m: .2});
  assert.ok(Math.abs(sourceFieldDistance([box], 0, [1.2, 0, 0])) < 1e-12);
  assert.equal(sourceFieldDistance([{...box, inside: true}], 0, [0, 0, 0]), 1.2);
  const cylinder = field({kind: 'cylinder', radius_m: 2, halfLength_m: 3, roundingRadius_m: .25});
  assert.equal(sourceFieldDistance([cylinder], 0, [2, 0, 0]), 0);
  assert.equal(sourceFieldDistance([cylinder], 0, [0, 3.25, 0]), 0);
  const rotated = {...box, axis: [0, 0, 1] as const, angle_rad: Math.PI / 2};
  assert.ok(Math.abs(sourceFieldDistance([rotated], 0, [0, 1.2, 0])) < 1e-12);
  const fields = [field({root: false}), field({index: 1, center_m: [1, 0, 0], root: false}), field({index: 2, kind: 'intersect', children: [0, 1]})];
  assert.equal(sourceFieldDistance(fields, 2, [0, 0, 0]), 0);
  assert.equal(sourceFieldDistance(fields, 2, [.5, 0, 0]), -.5);
  assert.equal(sourceTerminatesPoint([field({usage: 'terminator', inside: true})], [2, 0, 0]), true);
  assert.throws(() => sourceFieldDistance([field({kind: 'file', filename: 'missing.obj'})], 0, [0, 0, 0]), /unavailable|not loaded/);
  assert.throws(() => sourceFieldDistance([field({kind: 'intersect', children: [0]})], 0, [0, 0, 0]), /cyclic/);
});

test('emission emits initial/delayed events once, clips intervals and inherits first CSG child velocity', () => {
  const fields = [field({root: false, emission: [{start_s: 0, end_s: 0, velocity_m_s: [1, 0, 0]}, {start_s: 2, end_s: 2, velocity_m_s: [2, 0, 0]}, {start_s: 3, end_s: 4, velocity_m_s: [3, 0, 0]}]}),
    field({index: 1, root: false, emission: [{start_s: 0, end_s: 0, velocity_m_s: [9, 0, 0]}]}),
    field({index: 2, kind: 'intersect', children: [0, 1]})];
  const scheduler = new SourceEmissionScheduler(fields);
  assert.deepEqual(scheduler.advance(0, 1).map(e => e.velocity_m_s), [[1, 0, 0]]);
  assert.equal(scheduler.advance(1, 2).length, 0);
  assert.deepEqual(scheduler.advance(2, 3).map(e => e.velocity_m_s), [[2, 0, 0]]);
  const interval = scheduler.advance(3, 3.5)[0]!;
  assert.equal(interval.kind, 'interval'); assert.equal(interval.start_s, 3); assert.equal(interval.end_s, 3.5);
  assert.equal(scheduler.advance(3.5, 5)[0]!.end_s, 4);
  assert.throws(() => scheduler.advance(0, 1), /increasing/);
});

test('source motion applies the same cubic increment to supports and collision fields, and rolls back', async () => {
  const {SourceMotionRuntime, sourceCubicVelocity} = await import('./source-motion');
  const script = {kind: 'translate' as const, group: 1, start_s: 0, end_s: 1, vector: [1, 0, 0] as const, easing: 'cubic' as const};
  const topology = {restPositions_m: [0, 0, 0, 1, 0, 0], triangles: [], rodOffsets: [0], rodIndices: [], groups: [1, 1], fixed: [1, 2]};
  const motion = new SourceMotionRuntime([script], topology, [field({group: 1})]);
  const saved = motion.checkpoint();
  motion.step(.1, 0);
  const dx = sourceCubicVelocity(script, .1) * .1;
  assert.equal(motion.supportPositions_m[0], dx);
  assert.equal(motion.fields[0]!.center_m![0], dx);
  assert.equal(motion.supportPositions_m[3], 1, 'twist-only fixed flag does not prescribe position');
  assert.ok(Math.abs(sourceFieldDistance(motion.fields, 0, [1 + dx, 0, 0])) < 1e-12);
  const next = motion.checkpoint(); motion.restore(saved); assert.equal(motion.supportPositions_m[0], 0);
  motion.step(.1, 0); assert.deepEqual(motion.checkpoint(), next);
});

test('source local/global rotations compose in XML order and explicit pivots move colliders', async () => {
  const {SourceMotionRuntime} = await import('./source-motion');
  const topology = {restPositions_m: [2, 0, 0], triangles: [], rodOffsets: [0], rodIndices: [], groups: [0], fixed: [1]};
  const script = {kind: 'rotate' as const, group: 0, start_s: 0, end_s: 1, vector: [0, 0, 1] as const,
    angle_rad: Math.PI / 2, origin_m: [0, 0, 0] as const, easing: 'cubic' as const, easeIn_s: 0, easeOut_s: 0};
  const motion = new SourceMotionRuntime([script], topology, [field({center_m: [1, 0, 0]})]);
  motion.step(1);
  assert.ok(Math.abs(motion.supportPositions_m[0]!) < 1e-12);
  assert.ok(Math.abs(motion.supportPositions_m[1]! - 2) < 1e-12);
  assert.ok(Math.abs(motion.fields[0]!.center_m![1] - 1) < 1e-12);
  assert.ok(Math.abs(motion.groupRotationIncrement(0)[2] - Math.SQRT1_2) < 1e-12);
});

test('source definitions retain original geometry/material/scripts and explicit admission requirements', async () => {
  const {wetClothSourceDefinitions, prepareWetClothSource, buildWetClothSourceScene} = await import('./scenes/definitions');
  const {validateScene, serializeScene, parseScene} = await import('../core/model');
  assert.equal(wetClothSourceDefinitions.length, 36);
  const id = 'wetcloth-source-simple-cloth';
  await prepareWetClothSource(id, (async (url) => new Response(await readAsset(String(url)))) as typeof fetch);
  const scene = buildWetClothSourceScene(id);
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.cloth![0]!.scripts!.length, 12);
  assert.equal(scene.cloth![0]!.materials[0]!.youngModulus_Pa, 66000);
  assert.equal(scene.wetClothSource!.sourceId, id);
  assert.ok(scene.wetClothSource!.requiredCapabilities.includes('free-space-boundary'));
  assert.deepEqual(parseScene(serializeScene(scene)).wetClothSource, scene.wetClothSource);
  const bad = structuredClone(scene) as unknown as {wetClothSource: {fields: unknown[]}};
  bad.wetClothSource.fields = [field({kind: 'unknown' as never})];
  assert.ok(validateScene(bad as never).some(e => e.includes('Unsupported source field')));
});

test('reference sphere initializes real liquid with authored velocity; incompatible sources stay gated', async () => {
  const {sourceInitialLiquid} = await import('./scenes/initial-liquid');
  const a = field({emission: [{start_s: 0, end_s: 0, velocity_m_s: [0, -2, 0]}]});
  const exact = sourceInitialLiquid([a]);
  assert.deepEqual(exact.volumes, [{shape: 'sphere', center_m: {x: 0, y: 0, z: 0}, radius_m: 1}]);
  assert.deepEqual(exact.velocity_m_s, {x: 0, y: -2, z: 0});
  assert.equal(exact.needsSourceEmitter, false);
  const different = sourceInitialLiquid([a, field({index: 1, emission: [{start_s: 0, end_s: 0, velocity_m_s: [0, 1, 0]}]})]);
  assert.equal(different.needsSourceEmitter, true); assert.deepEqual(different.volumes, []);
  const round = sourceInitialLiquid([{...a, kind: 'box', halfExtents_m: [1, 1, 1], roundingRadius_m: .1}]);
  assert.equal(round.needsSourceEmitter, true); assert.deepEqual(round.volumes, []);
  const {prepareWetClothSource, buildWetClothSourceScene} = await import('./scenes/definitions');
  const id = 'wetcloth-source-pore-test-mid';
  await prepareWetClothSource(id, (async url => new Response(await readAsset(String(url)))) as typeof fetch);
  const scene = buildWetClothSourceScene(id);
  assert.equal(scene.fluid.initialLiquidVolumes!.length, 1);
  assert.equal(scene.fluid.initialLiquidVolumes![0]!.shape, 'sphere');
  assert.ok(!scene.wetClothSource!.requiredCapabilities.includes('source-sdf-emission'));
});

test('source allocation preserves original cell spacing and camera eye/target/lens', async () => {
  const {prepareWetClothSource, buildWetClothSourceScene} = await import('./scenes/definitions');
  const {sourceCameraForScene} = await import('./source-camera');
  const {sceneFluidFrame, sceneLatticeDimensions} = await import('../core/scene-lattice-dimensions');
  const {cameraPosition} = await import('../core/math');
  const source = wetClothSource('wetcloth-source-simple-cloth');
  await prepareWetClothSource(source.id, (async url => new Response(await readAsset(String(url)))) as typeof fetch);
  const scene = buildWetClothSourceScene(source.id);
  assert.ok(sceneLatticeDimensions(scene).every(n => n % 32 === 0));
  assert.ok(sceneFluidFrame(scene).cellSize_m.every(h => Math.abs(h - source.cellSize_m) < 1e-15));
  const view = sourceCameraForScene(scene)!, eye = cameraPosition(view.camera), center = view.camera.target_m;
  assert.ok(Math.abs(Math.hypot(eye.x-center.x,eye.y-center.y,eye.z-center.z)-source.camera!.distance_m)<1e-12);
  assert.equal(view.camera.tanHalfFov, Math.tan(source.camera!.fov_deg*Math.PI/360));
  assert.equal(center.y, source.camera!.center_m[1]+scene.wetClothSource!.sourceOffset_m[1]);
  assert.ok(Number.isFinite(view.roll_rad));
});

test('stationary delayed spheres use the real scheduled-drop path; moving emitters remain explicit', async () => {
  const {sourceInitialLiquid}=await import('./scenes/initial-liquid');
  const f=field({emission:[{start_s:2,end_s:2,velocity_m_s:[0,0,0]}]});
  const delayed=sourceInitialLiquid([f]);assert.equal(delayed.needsSourceEmitter,false);assert.equal(delayed.scheduledDrops[0]!.time_s,2);
  assert.equal(delayed.volumes.length,0);
  const script={kind:'translate' as const,group:0,start_s:0,end_s:1,easing:'cubic' as const,easeStart_s:1/3,easeEnd_s:1/3,global:false,displacement_m:[1,0,0] as const,sourceAttributes:{}};
  const moving=sourceInitialLiquid([f],[script]);assert.equal(moving.needsSourceEmitter,true);assert.equal(moving.scheduledDrops.length,0);
});

test('source physics ablations preserve parser defaults and explicit disabled solves',()=>{
  for(const source of wetClothWaterSources.filter(s=>s.counts.vertices)){
    const document=sourceClothDocument(source.id),settings=source.sourceSettings.liquid;
    assert.equal(document.physics?.solveSolid,Number(settings.solveSolid??1)!==0);
    assert.equal(document.physics?.varyingSolidFraction,Number(settings.useVaryingFraction??0)!==0);
    assert.equal(document.physics?.porePressure,Number(settings.applyPorePressureSolid??0)!==0);
    assert.equal(document.physics?.pressureOnSolid,Number(settings.applyPressureSolid??1)!==0);
    assert.equal(document.physics?.manifoldTransport,true);
    assert.equal(document.physics?.pressureOnManifold,Number(settings.applyPressureManifold??1)!==0);
    assert.equal(document.physics?.manifoldSubsteps,Number(source.sourceSettings.integrator.manifoldsubsteps??4));
    assert.equal(document.physics?.captureRate,Number(settings.elastoCaptureRate??1));
  }
});
