import type { SceneEvidenceCollectorRegistry } from "./scene-evidence-collector-runtime";
import { freeFallContactAttribution, freeFallOracle } from "./free-fall-contact-evidence";
import { auditHoseJetDrift } from "./webgpu-hose-drift";
import { inflowOutletCenter } from "../lib/inflow-boundary";

export const sceneEvidenceCollectorRegistry: SceneEvidenceCollectorRegistry = {
  "free-fall-contact-attribution": {
    id: "free-fall-contact-attribution",
    phase: "checkpoint",
    collect: ({ scene, grid, time_s, volumeField, velocityField }) => {
      if (!velocityField) throw new Error("free-fall contact attribution requires compact velocity evidence");
      const oracle = freeFallOracle(scene, grid);
      const gravity = scene.fluid.gravity_m_s2;
      return freeFallContactAttribution(volumeField, velocityField, grid,
        [gravity.x, gravity.y, gravity.z], time_s, oracle.centroidY_cells(time_s));
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
