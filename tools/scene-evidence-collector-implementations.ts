import type { SceneEvidenceCollectorRegistry } from "./scene-evidence-collector-runtime";
import { freeFallContactAttribution, freeFallOracle } from "./free-fall-contact-evidence";
import { auditHoseJetDrift } from "./webgpu-hose-drift";
import { inflowOutletCenter } from "../lib/inflow-boundary";
import { measureFluidSymmetry } from "../lib/fluid-symmetry-diagnostic";

export const sceneEvidenceCollectorRegistry: SceneEvidenceCollectorRegistry = {
  "rigid-coupling": {
    id: "rigid-coupling",
    phase: "checkpoint",
    collect: ({ rigidCouplingSnapshot, velocityField, volumeField, grid }) => {
      if (!rigidCouplingSnapshot) throw new Error("rigid coupling requires GPU body/exchange evidence");
      if (!velocityField) throw new Error("rigid coupling requires compact velocity evidence");
      let maximum = 0, cell = 0, component = 0;
      for (let index = 0; index < volumeField.length; index += 1) {
        if (!(volumeField[index]! > 0.5)) continue;
        for (let axis = 0; axis < 3; axis += 1) {
          const value = Math.abs(velocityField[3 * index + axis] ?? NaN);
          if (Number.isFinite(value) && value > maximum) {
            maximum = value; cell = index; component = axis;
          }
        }
      }
      const [nx, ny] = grid;
      const signed = velocityField[3 * cell + component] ?? NaN;
      return { ...rigidCouplingSnapshot, maximumLiquidComponentSpeed_m_s: maximum,
        maximumLiquidComponentLocation: {
          x: cell % nx, y: Math.floor(cell / nx) % ny, z: Math.floor(cell / (nx * ny)), component,
          signedVelocity_m_s: signed,
        } };
    },
  },
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
  "fluid-symmetry": {
    id: "fluid-symmetry",
    phase: "checkpoint",
    collect: ({ grid, time_s, volumeField, velocityField, pressureField,
      pressureRhsField, pressureDiagonalField, pressureSection63DiagonalField, pressureSection63CaseIdField,
      pressureInitialResidualField,
      pressureInitialPreconditionedField, pressureInitialPreconditionedImageField,
      pressurePreconditionerPreSmoothedField, pressurePreconditionerInnerResidualField,
      pressurePreconditionerZeroSmoothedField, pressurePreconditionerFirstOperatorImageField,
      pressurePreconditionerFirstSmoothedField,
      pressurePreconditionerInnerCorrectionField, pressurePreconditionerPostCorrectedField,
      topologyField }) => {
      if (!velocityField || !pressureField || !pressureRhsField || !pressureDiagonalField || !topologyField) {
        throw new Error("fluid symmetry requires compact velocity, pressure operator, and topology evidence");
      }
      return measureFluidSymmetry({
        time_s, grid, volume: volumeField, velocity: velocityField,
        pressure: pressureField, rhs: pressureRhsField, diagonal: pressureDiagonalField,
        section63Diagonal: pressureSection63DiagonalField,
        section63CaseId: pressureSection63CaseIdField,
        initialResidual: pressureInitialResidualField,
        initialPreconditioned: pressureInitialPreconditionedField,
        initialPreconditionedImage: pressureInitialPreconditionedImageField,
        preconditionerPreSmoothed: pressurePreconditionerPreSmoothedField,
        preconditionerZeroSmoothed: pressurePreconditionerZeroSmoothedField,
        preconditionerFirstOperatorImage: pressurePreconditionerFirstOperatorImageField,
        preconditionerFirstSmoothed: pressurePreconditionerFirstSmoothedField,
        preconditionerInnerResidual: pressurePreconditionerInnerResidualField,
        preconditionerInnerCorrection: pressurePreconditionerInnerCorrectionField,
        preconditionerPostCorrected: pressurePreconditionerPostCorrectedField,
        topology: topologyField,
        wallLiquidThreshold: 1e-4,
      });
    },
  },
};
