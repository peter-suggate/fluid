import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { BEACH_DURATION_S, BEACH_CELL_M, BEACH_WATERLINE_M, BEACH_WAVE_RAMP_S, BEACH_PISTON_AMPLITUDE_M, beachPistonPosition, beachPistonTrack, createBeachScene } from "../lib/core/beach-scene";
import { validateScene } from "../lib/core/model";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleRigidPoseTrack } from "../lib/core/prescribed-rigid-motion";
import { decodeMeshVolume, sampleMeshVolumeDistance } from "../lib/core/imported-assets/mesh-volume";
import { seaweedSkin } from "../lib/core/seaweed-render";
import { meshAssetHeader } from "../lib/core/imported-assets/asset-library";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";

test("Beach registers full NB-FLIP with a moving solid wave generator and Blender scenery", () => {
  const definition = getSceneDefinition("beach"), scene = createBeachScene();
  assert.deepEqual(validateScene(scene), []);
  assert.doesNotThrow(() => buildEnvironmentProxyCatalog(scene, "garden"), "Imported field programs must expand in metre units without unsupported scaling");
  assert.equal(definition.methodProfile?.methodId, "uniform-narrow-band-flip");
  assert.equal(definition.methodProfile?.overrides?.rigidCoupling, "on");
  assert.equal(definition.shelf, "Water landscapes");
  assert.equal(definition.renderProfile?.environmentRefinementDepth, 2);
  assert.equal(scene.fluid.initialHeightField, undefined, "No animated heightfield substitutes for fluid");
  assert.equal(scene.fluid.inflow, undefined, "A piston moves existing water without adding volume");
  assert.equal(scene.rigidBodies[0]!.motion, "static");
  assert.ok(scene.rigidBodies[0]!.poseTrack);
  const floating = scene.rigidBodies.filter(body => body.appearance === "seaweed");
  assert.equal(floating.length, 4);
  for (const body of floating) {
    assert.equal(body.motion, "dynamic");
    assert.ok(body.density_kg_m3 < scene.fluid.density_kg_m3);
    assert.equal(body.poseTrack, undefined, "Seaweed motion must come from fluid forces");
    assert.equal(body.prescribedMotion, undefined);
    assert.equal(body.renderHidden, undefined);
    assert.equal(seaweedSkin(body)?.shape, 9);
    assert.ok(body.dimensions_m.y >= 3 * BEACH_CELL_M, "A subcell wet proxy launches light clumps under pressure coupling");
    const submerged = BEACH_WATERLINE_M - (body.position_m.y - body.dimensions_m.y / 2);
    assert.ok(Math.abs(submerged / body.dimensions_m.y - body.density_kg_m3 / scene.fluid.density_kg_m3) < 1e-6, "Start at hydrostatic draft");
    const grid = scene.terrain!.grid!, radius = Math.hypot(body.dimensions_m.x, body.dimensions_m.z) / 2;
    for (const dx of [-radius,radius]) for (const dz of [-radius,radius]) {
      const ix = Math.round((body.position_m.x + dx - grid.origin_m.x) / grid.spacing_m);
      const iz = Math.round((body.position_m.z + dz - grid.origin_m.z) / grid.spacing_m);
      assert.ok(body.position_m.y - body.dimensions_m.y / 2 > grid.heights_m[iz * grid.size.nx + ix]! + BEACH_CELL_M, "Initial seaweed must clear the sloping bed");
    }
  }
  assert.equal(scene.scenery!.nodes.filter(node => node.kind === "field-program" && node.program.sampledVolume?.asset === "beach-palm").length, 3);
  assert.equal(meshAssetHeader("beach-palm").cellSize, .0125, "The native leaf detail must survive the deeper dry-scene SVO");
  assert.equal(meshAssetHeader("beach-rocks").cellSize, .04);
  assert.ok(scene.lighting!.environment!.cloudCoverage! > .2);
});

test("the offshore piston starts and ends smoothly and never crosses the tank wall", () => {
  const track = beachPistonTrack();
  assert.equal(beachPistonPosition(-1), -14.5);
  assert.equal(beachPistonPosition(BEACH_DURATION_S + 1), -14.5);
  let maxSpeed = 0;
  for (let i = 1; i <= BEACH_DURATION_S * 120; i++) {
    const t = i / 120, x = sampleRigidPoseTrack(track, t).position_m.x;
    assert.ok(x - .5 > -16 && x + .5 < -13.4);
    maxSpeed = Math.max(maxSpeed, Math.abs(x - sampleRigidPoseTrack(track, t - 1 / 120).position_m.x) * 120);
  }
  assert.ok(maxSpeed > .8 && maxSpeed < 1.6);
  assert.ok(Math.abs(beachPistonPosition(BEACH_WAVE_RAMP_S) + 14.5 - BEACH_PISTON_AMPLITUDE_M) < 1e-6, "The first full stroke must arrive within a second, not after two wave periods");
  assert.ok(Math.abs(beachPistonPosition(1 / 60) + 14.5) * 60 < .005);
  assert.ok(Math.abs(beachPistonPosition(BEACH_DURATION_S - 1 / 60) + 14.5) * 60 < .001);
});

test("Blender sand and the exported collision heightfield share the same shoreline", () => {
  const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL("../public/assets/beach/beach-sand.fvx.gz", import.meta.url))));
  const rocks = decodeMeshVolume(gunzipSync(readFileSync(new URL("../public/assets/beach/beach-rocks.fvx.gz", import.meta.url))));
  const grid = createBeachScene().terrain!.grid!;
  for (const [x,z] of [[-8,0],[0,-4],[5,0],[8,4],[13,-6],[2.5,5.375],[-1,7.25]] as const) {
    const ix = Math.round((x - grid.origin_m.x) / grid.spacing_m);
    const iz = Math.round((z - grid.origin_m.z) / grid.spacing_m);
    const y = grid.heights_m[iz * grid.size.nx + ix]!;
    const distance = (height: number) => Math.min(sampleMeshVolumeDistance(volume, [x,height,z]), sampleMeshVolumeDistance(rocks, [x,height,z]));
    assert.ok(Math.abs(distance(y)) < .045, `${x},${z} water/shore contact`);
    assert.ok(distance(y-.2) < 0);
  }
});

test("photographed rock colours survive GLB import and FVX baking", () => {
  const rock = meshAssetHeader("beach-rocks");
  const mean = rock.materials[0]!.baseColorLinear;
  assert.ok(mean.every(channel => channel > .005 && channel < .3), "A missing texture must not silently become the imported BSDF's grey .8 default");
  const distinct = new Set(rock.materials.map(material => material.baseColorLinear.join(",")));
  assert.ok(distinct.size > 20, "The baked stone must retain photographed mineral colour variation");
});
