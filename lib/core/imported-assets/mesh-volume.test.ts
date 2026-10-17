import assert from "node:assert/strict";
import test from "node:test";
import { ProceduralSolidField, validateProceduralSolid } from "../procedural-solid";
import { decodeMeshVolume, meshVolumeToSolid, sampleMeshVolume } from "./mesh-volume";

// An affine half-space with independently authored shading normals. Two bricks
// exercise interpolation, gradients and provenance across an actual seam.
function fixture(field = (x: number, y: number, _z: number) => (.25 * (x - 8) + y - 4) * .1) {
  const header = { version: 1, units: "metres", upAxis: "Y", asset: "inclined-plane",
    origin: [-.8, 0, -.4], dimensions: [16, 8, 8], cellSize: .1, bandWidth: 2,
    brickCount: 2, brickCells: 8, nodeStrideBytes: 16,
    materials: [{ name: "test", baseColorLinear: [.2, .4, .6], roughness: .5, metallic: 0 }] };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + json.length + 2 * (12 + 729 * 16)), view = new DataView(bytes.buffer);
  bytes.set(new TextEncoder().encode("FVX1")); view.setUint32(4, json.length, true); bytes.set(json, 8);
  let at = 8 + json.length;
  for (let b = 0; b < 2; b++) {
    view.setInt32(at, b * 8, true); at += 12;
    for (let i = 0; i < 729; i++, at += 16) {
      const x = i % 9 + b * 8, y = Math.floor(i / 9) % 9, z = Math.floor(i / 81);
      view.setFloat32(at, field(x, y, z), true);
      view.setUint32(at + 4, 0xe6669999, true); // approximately normalize(.25, 1, 0)
      view.setUint32(at + 8, 0x80008000, true); // authored +Z shading normal
      view.setUint32(at + 12, 1, true);
    }
  }
  return { bytes, headerBytes: json.length };
}

test("imported mesh keeps the signed field, derivative and authored shading separate", () => {
  const volume = decodeMeshVolume(fixture().bytes);
  for (const x of [-.01, 0, .01, .8]) {
    const sample = sampleMeshVolume(volume, [x, .35, 0]);
    assert.ok(Math.abs(sample.distance - (.25 * x - .05)) < 1e-7);
    assert.ok(Math.abs(sample.gradient[0] - .25) < 1e-6);
    assert.ok(Math.abs(sample.gradient[1] - 1) < 1e-6);
    assert.ok(Math.abs(sample.gradient[2]) < 1e-6);
    assert.ok(sample.shadingNormal[2] > .99999);
    assert.ok(sample.geometricNormal[1] > .96);
    assert.equal(sample.material, 1);
  }
  assert.equal(sampleMeshVolume(volume, [5, 0, 0]).material, 0);
  assert.throws(() => sampleMeshVolume(volume, [NaN, 0, 0]), /finite/);
});

test("a curved imported surface keeps subvoxel crossings and varying field normals", () => {
  const volume = decodeMeshVolume(fixture((x, y, z) => (Math.hypot(x - 8, y - 4, z - 4) - 3.25) * .1).bytes);
  const crossing = sampleMeshVolume(volume, [.325, .4, 0]);
  assert.ok(Math.abs(crossing.distance) < 1e-7, "the radius stays between lattice nodes");
  assert.ok(sampleMeshVolume(volume, [.31, .4, 0]).distance < 0);
  assert.ok(sampleMeshVolume(volume, [.34, .4, 0]).distance > 0);
  for (const direction of [[.6, .8, 0], [-.6, .8, 0], [0, .6, .8]]) {
    const position = [direction[0]! * .325, .4 + direction[1]! * .325, direction[2]! * .325] as const;
    const sample = sampleMeshVolume(volume, position), length = Math.hypot(...sample.gradient);
    const alignment = sample.gradient.reduce((sum, v, a) => sum + v * direction[a]!, 0) / length;
    assert.ok(alignment > .97, `curved gradient alignment ${alignment}`);
    assert.ok(Math.abs(sample.distance) < .012, "interpolation stays within its sampling error");
  }
});

test("mesh importer feeds the existing smooth solid field without replacing it with occupancy", () => {
  const volume = decodeMeshVolume(fixture().bytes), geometry = meshVolumeToSolid(volume, () => 7);
  validateProceduralSolid(geometry.solidBoundary);
  assert.ok(geometry.solidVoxels.length > 0);
  const field = new ProceduralSolidField(geometry.solidBoundary, []);
  for (const x of [7.9, 8, 8.1]) {
    const sample = field.sample([x, 4, 4])!;
    assert.ok(Math.abs(sample[3] - .25 * (x - 8)) < 1e-6);
    assert.ok(Math.abs(sample[0] - .25) < 1e-6);
    assert.ok(Math.abs(sample[1] - 1) < 1e-6);
  }
  assert.ok(field.open([8, 4, 4])! > .99);
  assert.ok(field.open([8, 2, 4])! < .01);
  assert.throws(() => meshVolumeToSolid(volume, () => 65536), /material IDs/);
});

test("mesh volume rejects corrupt payloads, invalid nodes and inconsistent shared faces", () => {
  const { bytes, headerBytes } = fixture();
  assert.throws(() => decodeMeshVolume(bytes.subarray(0, bytes.length - 1)), /payload length/);
  const invalid = bytes.slice(); new DataView(invalid.buffer).setFloat32(8 + headerBytes + 12, NaN, true);
  assert.throws(() => decodeMeshVolume(invalid), /node/);
  const seam = bytes.slice(); new DataView(seam.buffer).setFloat32(8 + headerBytes + 12 + 8 * 16, .123, true);
  assert.throws(() => decodeMeshVolume(seam), /shared/);
  const duplicate = bytes.slice(); new DataView(duplicate.buffer).setInt32(8 + headerBytes + 12 + 729 * 16, 0, true);
  assert.throws(() => decodeMeshVolume(duplicate), /duplicate/);
  const material = bytes.slice(); new DataView(material.buffer).setUint32(8 + headerBytes + 24, 99, true);
  assert.throws(() => decodeMeshVolume(material), /node/);
});
