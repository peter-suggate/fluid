import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createEmptyScene } from '../lib/core/empty-scene';
import { gesturePivot } from '../lib/core/camera-pivot';
import { sceneryPickAt } from '../lib/core/editor-scenery';
import { cameraPosition, zoomToward } from '../lib/core/math';
import { meshAssetHeader, meshAssetDistanceFieldReady, registerMeshAssetDistanceField } from '../lib/core/imported-assets/asset-library';
import { decodeMeshVolume } from '../lib/core/imported-assets/mesh-volume';

test('wheel pivot skips an unloaded paver and picks its real surface after the scalar handoff', () => {
  const asset = 'honed-travertine-paver', h = meshAssetHeader(asset);
  const center = h.origin.map((v, i) => v + h.dimensions[i]! * h.cellSize / 2);
  const scene = createEmptyScene({ extents_m: { x: 4, y: 4, z: 4 }, finestCellSize_m: .25 });
  scene.scenery = { palettes: {}, nodes: [
    { kind: 'room-shell', id: 'shell', materialModel: 'room', faces: [],
      floor: { colorLinear: [1, 1, 1] }, wall: { colorLinear: [1, 1, 1] }, ceiling: { colorLinear: [1, 1, 1] } },
    { kind: 'field-program', id: 'paver', place: { units: 'metres', position: { x: center[0]!, y: 1 + center[1]!, z: center[2]! } },
      program: { ops: [], result: 0, sampledVolume: { asset } }, material: { colorLinear: [1, 1, 1] } },
    { kind: 'box', id: 'ready-box', place: { units: 'metres', position: { x: 0, y: .3, z: 0 } },
      halfSize: { x: .2, y: .2, z: .2 }, material: { colorLinear: [1, 1, 1] } },
  ] };
  const ray = { origin: { x: 0, y: 3, z: 0 }, direction: { x: 0, y: -1, z: 0 } };
  const camera = { azimuth_rad: 0, elevation_rad: 1.3, distance_m: 3, target_m: { x: 0, y: .5, z: 0 } };
  const context = { scene, bodies: [], pickingAvailable: true };
  assert.equal(meshAssetDistanceFieldReady(asset), false);
  assert.equal(sceneryPickAt(scene, ray), 'ready-box', 'available geometry remains pickable');
  const before = gesturePivot(context, camera, ray);
  assert.ok(Object.values(before).every(Number.isFinite));
  assert.ok(Object.values(cameraPosition(zoomToward(camera, before, -100))).every(Number.isFinite));

  const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL('../public/assets/pool-party/honed-travertine-paver.fvx.gz', import.meta.url))), true);
  registerMeshAssetDistanceField(asset, { header: volume.header, bricks: new Map([...volume.bricks].map(
    ([key, { origin, distance }]) => [key, { origin, distance }])) });
  assert.equal(sceneryPickAt(scene, ray), 'paver', 'same cached catalog recovers without a scene reset');
  const after = gesturePivot(context, camera, ray);
  assert.ok(Math.abs(after.y - 1.03) < .005, `pivot is on the actual stone: ${after.y}`);
});
