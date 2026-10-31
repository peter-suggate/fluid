import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { initialLiquidVertexPhi } from "../lib/core/initial-liquid-lattice";
import { NBFLIP_STAIR_CUBE_ID, NBFLIP_STAIR_VESSEL_POSITION as P } from "../lib/core/nbflip-stair-cube-scene";
import { getScenePreset } from "../lib/core/scenes";
import { sceneShape } from "../lib/core/scene-shape";
import { interactionScene } from "../lib/interaction/adapters/scene";

// The t=0 phi continues through a body's contact face (Batty, Bertails and
// Bridson 2007; Ng, Min and Gibou 2009): the fluid measures a cell as its open
// fraction times phi's fill, so liquid authored up to a wall that cuts a cell
// must fill that cell's open part, not a share of it.
test("liquid authored to a sealed vessel's off-grid walls measures its authored volume at t=0", () => {
  const base = getScenePreset(NBFLIP_STAIR_CUBE_ID).create();
  // The vessel's whole interior: its y and z walls cut cells (1.98 and 3.10 m, 48.25 and 80.25 cells).
  base.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: P.x-0.64, y: P.y-0.56, z: P.z-0.64 }, max_m: { x: P.x+0.64, y: P.y+0.56, z: P.z+0.64 } }];
  const scene = interactionScene(base), c = scene.container, h = 0.04;
  const [nx, ny, nz] = [Math.round(c.width_m/h), Math.round(c.height_m/h), Math.round(c.depth_m/h)];
  const phi = initialLiquidVertexPhi(scene, [nx, ny, nz]), X = nx+1, Y = ny+1;
  const body = scene.rigidBodies.find(candidate => candidate.fluidBoundary === "interior")!, shape = sceneShape(body.shape);
  // The fluid's body solid: eight samples at 0.4 of the cell about its centre.
  const open = (x: number, y: number, z: number) => {
    let inside = 0;
    for (let s=0; s<8; s++) inside += Number(shape.inside(body.dimensions_m, {
      x: ((x+0.5+(s&1 ? 0.4 : -0.4))/nx-0.5)*c.width_m-body.position_m.x,
      y: (y+0.5+(s&2 ? 0.4 : -0.4))/ny*c.height_m-body.position_m.y,
      z: ((z+0.5+(s&4 ? 0.4 : -0.4))/nz-0.5)*c.depth_m-body.position_m.z }));
    return inside/8;
  };
  let filled = 0, cutOpen = 0;
  for (let z=40; z<90; z++) for (let y=40; y<90; y++) for (let x=84; x<132; x++) {
    let negative = 0;
    for (let s=0; s<8; s++) negative += Number(phi[x+(s&1)+X*(y+(s>>1&1)+Y*(z+(s>>2&1)))]! < 0);
    if (negative === 8) filled += open(x, y, z); else if (negative > 0) cutOpen += open(x, y, z);
  }
  // Every open cell is wholly inside phi, and no open cell holds a surface.
  assert.equal(filled, 32*28*32);
  assert.equal(cutOpen, 0);
});
