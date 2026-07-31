import type { SceneEvidenceCollectorRegistry } from "./scene-evidence-collector-runtime";
import { freeFallContactAttribution, freeFallOracle } from "./free-fall-contact-evidence";
import { auditHoseJetDrift } from "./webgpu-hose-drift";
import { inflowOutletCenter } from "../lib/inflow-boundary";

export const sceneEvidenceCollectorRegistry: SceneEvidenceCollectorRegistry = {
  "free-fall-contact-attribution": {
    id: "free-fall-contact-attribution",
    phase: "checkpoint",
    collect: ({ scene, grid, time_s, volumeField, velocityField, fineUpperSurfaceField }) => {
      if (!velocityField) throw new Error("free-fall contact attribution requires compact velocity evidence");
      if (!fineUpperSurfaceField) throw new Error("free-fall contact attribution requires fine upper-surface evidence");
      const oracle = freeFallOracle(scene, grid);
      const gravity = scene.fluid.gravity_m_s2;
      const seed = scene.fluid.initialBrickSeeds_m?.[0];
      const seedFootprint = seed ? {
        originX: Math.floor(Math.min(grid[0] - 1, Math.max(0,
          Math.floor((seed.x / scene.container.width_m + 0.5) * grid[0]))) / 8) * 8,
        originZ: Math.floor(Math.min(grid[2] - 1, Math.max(0,
          Math.floor((seed.z / scene.container.depth_m + 0.5) * grid[2]))) / 8) * 8,
        size: 8,
      } : undefined;
      return freeFallContactAttribution(volumeField, velocityField, grid,
        [gravity.x, gravity.y, gravity.z], time_s, oracle.centroidY_cells(time_s), seedFootprint,
        fineUpperSurfaceField);
    },
  },
  "hose-jet-drift": {
    id: "hose-jet-drift",
    phase: "terminal",
    collect: ({ scene, grid, volumeField, velocityField }) => {
      const inflow = scene.fluid.inflow;
      if (!inflow || !velocityField) {
        throw new Error("hose jet drift requires declared inflow geometry and compact velocity evidence");
      }
      return auditHoseJetDrift(volumeField, velocityField, grid, scene.container, inflow,
        inflowOutletCenter(inflow), scene.fluid.gravity_m_s2);
    },
  },
  "collocated-velocity": {
    id: "collocated-velocity",
    phase: "terminal",
    collect: ({ volumeField, velocityField, compactVelocityEvidence }) => {
      if (!velocityField) throw new Error("collocated velocity requires a velocity field");
      return { field: velocityField, volume: volumeField, compactRaster: compactVelocityEvidence };
    },
  },
};
