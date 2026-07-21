import assert from "node:assert/strict";
import test from "node:test";
import {
  constructOctreePowerCell,
  createOctreePowerSite,
  matchSharedOctreePowerFace,
  powerBoxBoundary,
  serializeOctreePowerCell,
  type OctreePowerSite,
} from "../lib/octree-power-geometry";
import { OCTREE_CUBE_TRANSFORMS, transformPowerVector } from "../lib/octree-power-topology";

const close = (actual: number, expected: number, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);

function uniformSites(): OctreePowerSite[] {
  return [
    createOctreePowerSite("anchor", [0, 0, 0], 1),
    createOctreePowerSite("x-", [-1, 0, 0], 1), createOctreePowerSite("x+", [1, 0, 0], 1),
    createOctreePowerSite("y-", [0, -1, 0], 1), createOctreePowerSite("y+", [0, 1, 0], 1),
    createOctreePowerSite("z-", [0, 0, -1], 1), createOctreePowerSite("z+", [0, 0, 1], 1),
  ];
}

test("uniform weighted site reconstructs the ordinary Cartesian cube", () => {
  const sites = uniformSites();
  const cell = constructOctreePowerCell(sites[0], sites);
  assert.equal(cell.faces.length, 6);
  assert.equal(cell.vertices.length, 8);
  close(cell.volume, 1);
  cell.centroid.forEach((value) => close(value, 0.5));
  for (const face of cell.faces) {
    close(face.area, 1);
    close(face.dualDistance, 1);
    assert.equal(face.normal.filter((value) => Math.abs(value) > 1e-12).length, 1);
  }
});

test("coarse/fine power patch partitions volume and creates oblique faces", () => {
  const sites: OctreePowerSite[] = [createOctreePowerSite("coarse", [0, 0, 0], 2)];
  for (let z = 0; z < 3; z += 1) for (let y = 0; y < 3; y += 1) for (let x = 0; x < 3; x += 1) {
    if (x < 2 && y < 2 && z < 2) continue;
    sites.push(createOctreePowerSite(`${x}${y}${z}`, [x, y, z], 1));
  }
  const boundaries = powerBoxBoundary([0, 0, 0], [3, 3, 3]);
  const cells = sites.map((site) => constructOctreePowerCell(site, sites, boundaries));
  close(cells.reduce((sum, cell) => sum + cell.volume, 0), 27, 2e-8);
  const coarse = cells[0];
  assert.ok(coarse.faces.some((face) => face.kind === "site"
    && face.normal.filter((value) => Math.abs(value) > 1e-8).length === 3), "transition must not remain axis-only");
  assert.ok(coarse.faces.every((face) => face.area > 0 && Number.isFinite(face.area)));
});

test("3D octree edge neighbors can acquire a reciprocal power face", () => {
  // Four octree cells incident on an x-aligned edge. A and D only touch along
  // that original edge; the unequal radii make their diagonal power face live.
  const sites = [
    createOctreePowerSite("A", [0, -2, -2], 2),
    createOctreePowerSite("B", [0, 0, -1], 1),
    createOctreePowerSite("C", [0, -1, 0], 1),
    createOctreePowerSite("D", [0, 0, 0], 1),
  ];
  const boundaries = powerBoxBoundary([0, -2, -2], [2, 2, 2]);
  const match = matchSharedOctreePowerFace(sites[0], sites[3], sites, boundaries);
  assert.ok(match.negative.area > 0);
  close(match.areaError, 0, 1e-9);
  close(match.centroidError, 0, 1e-9);
  close(match.normalError, 0, 1e-9);
});

test("representative power cell is invariant under all rotations and reflections", () => {
  const source = uniformSites();
  const anchorCenter = source[0].center;
  for (const transform of OCTREE_CUBE_TRANSFORMS) {
    const sites = source.map((site) => {
      const relative = site.center.map((value, axis) => value - anchorCenter[axis]) as [number, number, number];
      const center = transformPowerVector(relative, transform).map((value, axis) => value + anchorCenter[axis]);
      return createOctreePowerSite(site.key, center.map((value) => value - site.size / 2) as [number, number, number], site.size);
    });
    const cell = constructOctreePowerCell(sites[0], sites);
    close(cell.volume, 1);
    assert.equal(cell.faces.length, 6);
  }
});

test("shared faces agree from both incident cells and output is deterministic", () => {
  const sites = uniformSites();
  const match = matchSharedOctreePowerFace(sites[0], sites[2], sites, powerBoxBoundary([-1, -1, -1], [2, 2, 2]));
  close(match.areaError, 0);
  close(match.centroidError, 0);
  close(match.normalError, 0);
  const outputs = Array.from({ length: 5 }, () => serializeOctreePowerCell(constructOctreePowerCell(sites[0], [...sites].reverse())));
  assert.equal(new Set(outputs).size, 1);
});

test("oracle rejects empty/unbounded cells with a readable configuration dump", () => {
  const anchor = createOctreePowerSite("lonely", [0, 0, 0], 1);
  assert.throws(() => constructOctreePowerCell(anchor, [anchor]), /"anchor": "lonely"/);
});
