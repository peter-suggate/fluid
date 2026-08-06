import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_BRICK_DRY_EMPTY_IDENTITY,
  SPARSE_BRICK_DRY_PAYLOAD,
  censusSparseBrickDryPayload,
  classifySparseBrickDryBrick,
  packSparseBrickDryBrick,
  sparseBrickDryOccupied,
  sparseBrickDryVoxelSolid,
  unpackSparseBrickDryVoxel,
} from "../lib/sparse-brick-dry-payload";
import {
  SPARSE_BRICK_NO_OWNER,
  SPARSE_BRICK_PAYLOAD_PROFILES,
  packMaterialOwner,
  resolveSparseBrickPayloadLayout,
} from "../lib/sparse-brick-octree";
import {
  sparseSceneProxyVoxelizationShader,
  sparseSceneProxyVoxelizationShaderFor,
} from "../lib/webgpu-sparse-scene-proxies";
import {
  webgpuSvoBrickOccupancyBuildWGSL,
  webgpuSvoBrickOccupancyBuildWGSLFor,
} from "../lib/webgpu-svo-brick-occupancy";
import {
  liveSvoDerivedBuildWGSL,
  liveSvoDerivedBuildWGSLFor,
  liveSvoRadianceFeedbackWGSL,
  liveSvoRadianceFeedbackWGSLFor,
} from "../lib/webgpu-svo-live-derived-builder";

const { voxelsPerBrick, headerBytes, occupancyBytes, paletteBytes, indexBytes } = SPARSE_BRICK_DRY_PAYLOAD;

function emptyBrick(): Uint32Array {
  return new Uint32Array(voxelsPerBrick).fill(SPARSE_BRICK_DRY_EMPTY_IDENTITY);
}

/** Round-trip every voxel of a brick through pack/unpack. */
function assertRoundTrip(identities: ArrayLike<number>, voxelOffset = 0): void {
  const { words } = packSparseBrickDryBrick(identities, voxelOffset);
  for (let local = 0; local < voxelsPerBrick; local += 1) {
    const source = (identities[voxelOffset + local] ?? 0) >>> 0;
    const expected = sparseBrickDryVoxelSolid(source) ? source : SPARSE_BRICK_DRY_EMPTY_IDENTITY;
    assert.equal(unpackSparseBrickDryVoxel(words, local), expected, `voxel ${local}`);
    assert.equal(sparseBrickDryOccupied(words, local), sparseBrickDryVoxelSolid(source), `occupancy ${local}`);
  }
}

test("air and never-voxelized zero both read empty", () => {
  // Two distinct encodings mean "no surface here" and conflating either with a
  // real material would draw the whole domain solid.
  assert.equal(sparseBrickDryVoxelSolid(SPARSE_BRICK_DRY_EMPTY_IDENTITY), false);
  assert.equal(sparseBrickDryVoxelSolid(0), false);
  assert.equal(sparseBrickDryVoxelSolid(packMaterialOwner(2, SPARSE_BRICK_NO_OWNER)), true);
  // An owner alone is not solidity: a rigid body's owner over material zero is
  // still air, which is exactly the case webgpu-svo-dry-scene.ts:4944 tests.
  assert.equal(sparseBrickDryVoxelSolid(packMaterialOwner(0, 7)), false);
});

test("an empty brick costs only its header", () => {
  const classification = classifySparseBrickDryBrick(emptyBrick());
  assert.equal(classification.kind, "empty");
  assert.equal(classification.occupiedVoxelCount, 0);
  assert.equal(classification.byteLength, headerBytes);
  assertRoundTrip(emptyBrick());
});

test("a single-owner brick packs to a header plus a 512-bit mask", () => {
  const terrain = packMaterialOwner(2, 1024);
  const identities = emptyBrick();
  // A half-filled brick: the shape the ground makes in a surface brick.
  for (let local = 0; local < voxelsPerBrick; local += 1) if ((local >>> 3) % 8 < 4) identities[local] = terrain;
  const classification = classifySparseBrickDryBrick(identities);
  assert.equal(classification.kind, "uniform");
  assert.deepEqual(classification.palette, [terrain]);
  assert.equal(classification.occupiedVoxelCount, 256);
  assert.equal(classification.byteLength, headerBytes + occupancyBytes);
  // 2048 bytes today for the same 512 voxels.
  assert.equal(voxelsPerBrick * 4 / classification.byteLength, 2048 / 72);
  assertRoundTrip(identities);
});

test("a two-to-four owner brick takes the palette layout", () => {
  const owners = [packMaterialOwner(2, 1024), packMaterialOwner(34, 7), packMaterialOwner(35, 8), packMaterialOwner(36, 9)];
  for (const width of [2, 3, 4]) {
    const identities = emptyBrick();
    for (let local = 0; local < voxelsPerBrick; local += 1) identities[local] = owners[local % width];
    const classification = classifySparseBrickDryBrick(identities);
    assert.equal(classification.kind, "palette", `width ${width}`);
    assert.equal(classification.palette.length, width);
    assert.equal(classification.byteLength, headerBytes + occupancyBytes + paletteBytes + indexBytes);
    assertRoundTrip(identities);
  }
});

test("the palette preserves which owner each voxel had, not just how many", () => {
  // The failure this guards is a palette that round-trips the *count* while
  // permuting slots, which would silently repaint one prop with another's
  // material and never trip an occupancy check.
  const owners = [packMaterialOwner(34, 7), packMaterialOwner(35, 8), packMaterialOwner(36, 9)];
  const identities = emptyBrick();
  for (let local = 0; local < voxelsPerBrick; local += 1) {
    if (local % 5 === 0) continue;
    identities[local] = owners[(local * 7) % owners.length];
  }
  const { words } = packSparseBrickDryBrick(identities);
  for (let local = 0; local < voxelsPerBrick; local += 1) {
    if (local % 5 === 0) { assert.equal(unpackSparseBrickDryVoxel(words, local), SPARSE_BRICK_DRY_EMPTY_IDENTITY); continue; }
    assert.equal(unpackSparseBrickDryVoxel(words, local), owners[(local * 7) % owners.length]);
  }
});

test("a fifth owner overflows to a retained per-voxel lane rather than losing an identity", () => {
  const identities = emptyBrick();
  for (let local = 0; local < voxelsPerBrick; local += 1) identities[local] = packMaterialOwner(32 + (local % 5), local % 5);
  const classification = classifySparseBrickDryBrick(identities);
  assert.equal(classification.kind, "overflow");
  assert.equal(classification.palette.length, SPARSE_BRICK_DRY_PAYLOAD.maximumPaletteEntries);
  assert.equal(classification.byteLength, headerBytes + voxelsPerBrick * 4);
  // Lossless, which is the whole point of the exception: the fifth identity
  // survives even though the palette could not hold it.
  assertRoundTrip(identities);
});

test("the exception path is byte-for-byte no worse than today plus a header", () => {
  const identities = emptyBrick();
  for (let local = 0; local < voxelsPerBrick; local += 1) identities[local] = packMaterialOwner(32 + (local % 9), local % 9);
  const classification = classifySparseBrickDryBrick(identities);
  assert.equal(classification.byteLength - headerBytes, voxelsPerBrick * 4);
});

test("brick-local addressing is respected at a non-zero voxel offset", () => {
  const lane = new Uint32Array(voxelsPerBrick * 3).fill(SPARSE_BRICK_DRY_EMPTY_IDENTITY);
  const identity = packMaterialOwner(17, 3);
  for (let local = 0; local < voxelsPerBrick; local += 1) if (local % 3 === 0) lane[voxelsPerBrick + local] = identity;
  const classification = classifySparseBrickDryBrick(lane, voxelsPerBrick);
  assert.equal(classification.kind, "uniform");
  assert.equal(classification.occupiedVoxelCount, Math.ceil(voxelsPerBrick / 3));
  assertRoundTrip(lane, voxelsPerBrick);
  // The neighbouring bricks must not have been read.
  assert.equal(classifySparseBrickDryBrick(lane, 0).kind, "empty");
  assert.equal(classifySparseBrickDryBrick(lane, voxelsPerBrick * 2).kind, "empty");
});

test("a short lane is refused rather than read past its end", () => {
  assert.throws(() => classifySparseBrickDryBrick(new Uint32Array(voxelsPerBrick - 1)), RangeError);
  assert.throws(() => classifySparseBrickDryBrick(new Uint32Array(voxelsPerBrick), 1), RangeError);
  assert.throws(() => unpackSparseBrickDryVoxel(new Uint32Array(4), voxelsPerBrick), RangeError);
});

test("the census reproduces the measured hero-garden distribution", () => {
  // 9740 single-owner, 798 two, 85 three, 2 four, 2255 empty — the 6.25 mm
  // census in tmp/owner-census.ts. Scaled down by 20 to keep the test quick,
  // the *shape* of the answer is what is being pinned.
  const counts = { empty: 113, one: 487, two: 40, three: 4, four: 1 };
  const brickCount = counts.empty + counts.one + counts.two + counts.three + counts.four;
  const lane = new Uint32Array(brickCount * voxelsPerBrick).fill(SPARSE_BRICK_DRY_EMPTY_IDENTITY);
  let brick = 0;
  const fill = (bricks: number, width: number) => {
    for (let index = 0; index < bricks; index += 1, brick += 1) {
      for (let local = 0; local < voxelsPerBrick; local += 1) {
        lane[brick * voxelsPerBrick + local] = packMaterialOwner(32 + (local % width), local % width);
      }
    }
  };
  brick = counts.empty;
  fill(counts.one, 1); fill(counts.two, 2); fill(counts.three, 3); fill(counts.four, 4);

  const census = censusSparseBrickDryPayload(lane, brickCount);
  assert.equal(census.brickCount, brickCount);
  assert.equal(census.emptyBricks, counts.empty);
  assert.equal(census.uniformBricks, counts.one);
  assert.equal(census.paletteBricks, counts.two + counts.three + counts.four);
  assert.equal(census.overflowBricks, 0);
  assert.deepEqual(census.paletteHistogram, { 1: counts.one, 2: counts.two, 3: counts.three, 4: counts.four });
  // Every brick is represented losslessly and the lane shrinks by ~24x.
  assert.equal(census.perVoxelBytes, brickCount * voxelsPerBrick * 4);
  assert.ok(census.compressionRatio > 20, `ratio ${census.compressionRatio}`);
  assert.ok(census.meanBytesPerBrick < 90, `mean ${census.meanBytesPerBrick} B/brick`);
});

test("the dry profile drops the three lanes a solverless world never writes", () => {
  const full = SPARSE_BRICK_PAYLOAD_PROFILES.full.map((lane) => lane.name);
  const dry = SPARSE_BRICK_PAYLOAD_PROFILES.dry.map((lane) => lane.name);
  assert.deepEqual(full, ["geometry", "velocity", "materialOwners", "sceneGeometry", "sceneMaterialOwners"]);
  assert.deepEqual(dry, ["sceneGeometry", "sceneMaterialOwners"]);
});

test("the dry geometry lane keeps exactly the two channels the voxeliser writes", () => {
  const [sceneGeometry] = SPARSE_BRICK_PAYLOAD_PROFILES.dry;
  // lib/webgpu-sparse-scene-proxies.ts:1972-1973 writes channels 1 and 2 only;
  // fluidSignedDistance and pressure have no writer on this lane at all.
  assert.deepEqual([...sceneGeometry.channels], ["solidSignedDistance", "solidFraction"]);
  assert.equal(sceneGeometry.strideBytes, 8);
  // solidSignedDistance must survive: it is the only smooth-normal source, and
  // safeNormal (webgpu-svo-live-derived-builder.ts:246) differentiates it.
  assert.ok(sceneGeometry.channels.includes("solidSignedDistance"));
});

test("resolved lane offsets are 256-aligned and the full profile is unchanged", () => {
  const voxels = 512 * 584;
  const full = resolveSparseBrickPayloadLayout("full", voxels);
  assert.equal(full.bytesPerVoxel, 56);
  assert.equal(full.lanes.geometry.offsetBytes, 0);
  assert.equal(full.lanes.velocity.offsetBytes, voxels * 16);
  assert.equal(full.lanes.materialOwners.offsetBytes, voxels * 32);
  assert.equal(full.lanes.sceneGeometry.offsetBytes, voxels * 36);
  assert.equal(full.lanes.sceneMaterialOwners.offsetBytes, voxels * 52);
  assert.equal(full.totalBytes, voxels * 56);
  for (const lane of Object.values(full.lanes)) assert.equal(lane.offsetBytes % 256, 0, lane.name);
});

test("the dry profile reserves an absent-lane page instead of aliasing offset zero", () => {
  const voxels = 512 * 584;
  const dry = resolveSparseBrickPayloadLayout("dry", voxels);
  assert.equal(dry.bytesPerVoxel, 12);
  // Absent lanes must not land on a real lane: reads there would silently
  // return scene geometry reinterpreted as velocity.
  for (const name of ["geometry", "velocity", "materialOwners"] as const) {
    assert.equal(dry.lanes[name].present, false);
    assert.equal(dry.lanes[name].offsetBytes, 0);
  }
  assert.equal(dry.lanes.sceneGeometry.present, true);
  assert.equal(dry.lanes.sceneGeometry.offsetBytes, 256);
  assert.equal(dry.lanes.sceneMaterialOwners.offsetBytes, 256 + voxels * 8);
  assert.equal(dry.totalBytes, 256 + voxels * 12);
  assert.equal(resolveSparseBrickPayloadLayout("full", voxels).totalBytes / dry.totalBytes > 4.6, true);
});

test("the dry shader reads no lane the dry profile does not allocate", () => {
  const dry = liveSvoDerivedBuildWGSLFor("dry") + liveSvoRadianceFeedbackWGSLFor("dry");
  // `payload[voxel*4u+N]` is the dynamic geometry lane addressed implicitly at
  // offset zero, and laneOffsets .x/.y are velocity and dynamic material. On a
  // dry world all three land in the 256-byte absent page and then run off it
  // into scene geometry, so not one of them may survive templating.
  assert.doesNotMatch(dry, /payload\[voxel\*4u/, "dynamic geometry lane is still read");
  assert.doesNotMatch(dry, /params\.laneOffsets\.x\+voxel/, "velocity lane is still read");
  assert.doesNotMatch(dry, /params\.laneOffsets\.y\+voxel/, "dynamic material lane is still read");
  // The scene lanes are the two that survive, at the narrowed stride.
  assert.match(dry, /params\.laneOffsets\.z\+voxel\*2u\+0u/, "scene solidSignedDistance");
  assert.match(dry, /params\.laneOffsets\.z\+voxel\*2u\+1u/, "scene solidFraction");
  assert.match(dry, /params\.laneOffsets\.w\+voxel/, "scene identity");
  assert.doesNotMatch(dry, /laneOffsets\.z\+voxel\*4u/, "scene geometry is still four channels wide");
});

test("solidSignedDistance still reaches safeNormal on a dry world", () => {
  // The one channel that must survive: safeNormal central-differences it, and
  // it is the only source of a normal that is not one of six cube faces.
  for (const profile of ["full", "dry"] as const) {
    const source = liveSvoDerivedBuildWGSLFor(profile);
    const solidDistance = source.match(/fn solidDistance\([^)]*\)->f32\{[^}]*\}/)?.[0];
    assert.ok(solidDistance, `${profile}: solidDistance is missing`);
    assert.match(solidDistance, /params\.laneOffsets\.z/, `${profile}: no scene distance read`);
    assert.match(source, /let normal=safeNormal\(leaf,sampleLocal\)/, `${profile}: safeNormal is unused`);
  }
});

test("the full expansion is unchanged by the introduction of profiles", () => {
  // Every existing GPU lane compiles the `full` text. If templating perturbed
  // one character the frame changes for a reason nobody asked for.
  assert.equal(liveSvoDerivedBuildWGSLFor("full"), liveSvoDerivedBuildWGSL);
  assert.equal(liveSvoRadianceFeedbackWGSLFor("full"), liveSvoRadianceFeedbackWGSL);
  assert.match(liveSvoDerivedBuildWGSL,
    /return min\(bitcast<f32>\(payload\[voxel\*4u\+1u\]\),bitcast<f32>\(payload\[params\.laneOffsets\.z\+voxel\*4u\+1u\]\)\)/);
  assert.match(liveSvoDerivedBuildWGSL, /let dynamicIdentity=payload\[params\.laneOffsets\.y\+voxel\]/);
  assert.match(liveSvoDerivedBuildWGSL, /let fluid=clamp\(bitcast<f32>\(payload\[params\.laneOffsets\.x\+voxel\*4u\+3u\]\),0\.,1\.\)/);
});

test("the voxeliser and occupancy passes narrow with the profile and are unchanged on full", () => {
  assert.equal(sparseSceneProxyVoxelizationShaderFor("full"), sparseSceneProxyVoxelizationShader);
  assert.equal(webgpuSvoBrickOccupancyBuildWGSLFor("full"), webgpuSvoBrickOccupancyBuildWGSL);
  // The voxeliser writes channels 1 and 2 of a four-wide lane on `full` and
  // channels 0 and 1 of a two-wide lane on `dry` — same two channels, and the
  // dead fluidSignedDistance/pressure slots simply stop existing.
  assert.match(sparseSceneProxyVoxelizationShader, /let geometryBase = sceneGeometryOffset\(\) \+ output \* 4u;/);
  assert.match(sparseSceneProxyVoxelizationShader, /payload\[geometryBase\+1u\]=bitcast<u32>\(bestDistance\)/);
  const dryProxy = sparseSceneProxyVoxelizationShaderFor("dry");
  assert.match(dryProxy, /let geometryBase = sceneGeometryOffset\(\) \+ output \* 2u;/);
  assert.match(dryProxy, /payload\[geometryBase\]=bitcast<u32>\(bestDistance\)/);
  assert.match(dryProxy, /payload\[geometryBase\+1u\]=bitcast<u32>\(primitiveFraction\)/);
  // The dynamic material word is control[18], which on a dry world addresses
  // the absent-lane page. Both passes that union it must stop reading it.
  assert.doesNotMatch(dryProxy, /controlLoad\(18u\)\+voxelOffset/);
  assert.doesNotMatch(webgpuSvoBrickOccupancyBuildWGSLFor("dry"), /let fluidMaterial=payload\[payloadIndex\]/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /let fluidMaterial=payload\[payloadIndex\]&0xffffu/);
});

test("the container-glass opacity exclusion survives both profiles", () => {
  // Excluded so the cone hierarchy does not turn the vessel into a projected
  // cutout. It is a no-op on the garden (no vessel), but the tank scenes that
  // need it are all `full` worlds, and the guard must not be profile-gated.
  for (const profile of ["full", "dry"] as const) {
    assert.match(liveSvoDerivedBuildWGSLFor(profile), /sceneIdentity&0xffffu\)==1u/, profile);
  }
});

test("an unknown profile is refused", () => {
  assert.throws(
    () => resolveSparseBrickPayloadLayout("sparse" as never, 512),
    /Unknown sparse brick payload profile/,
  );
});
