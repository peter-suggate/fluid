export interface AdaptiveSurfaceBreakupSample {
  readonly step: number;
  readonly topologyEpoch: number;
  readonly conservedMass_m3: number;
  readonly handoffDrift_m3: number;
  /** Volume implied directly by the cell-centred rho authority. */
  readonly massVisibleVolume_m3: number;
  /** Volume after rho -> nodal phi reconstruction and redistance. */
  readonly reconstructedVolume_m3: number;
  /** Optional final renderer measurement. */
  readonly renderedVolume_m3?: number;
  readonly massComponents: number;
  readonly reconstructedComponents: number;
  readonly renderedComponents?: number;
}

export interface AdaptiveSurfaceBreakupFailure {
  readonly step: number;
  readonly stage: "handoff" | "mass" | "reconstruction" | "renderer";
  readonly message: string;
}

export interface AdaptiveSurfaceBreakupTolerance {
  readonly mass_m3: number;
  readonly volume_m3: number;
}

/**
 * Attribute surface breakup to the first representation where it appears.
 *
 * The conserved rho field, reconstructed/redistanced phi, and renderer are
 * deliberately checked separately. This prevents an exact mass receipt from
 * masking a broken visible surface, and prevents a renderer defect from being
 * reported as transport loss.
 */
export function diagnoseAdaptiveSurfaceBreakup(
  samples: readonly AdaptiveSurfaceBreakupSample[],
  tolerance: AdaptiveSurfaceBreakupTolerance,
): readonly AdaptiveSurfaceBreakupFailure[] {
  if (samples.length === 0) return [];
  const failures: AdaptiveSurfaceBreakupFailure[] = [];
  const initial = samples[0]!;
  let previous = initial;
  for (const sample of samples) {
    if (sample.step < previous.step) throw new RangeError("adaptive surface samples must be step ordered");
    if (Math.abs(sample.conservedMass_m3 - initial.conservedMass_m3) > tolerance.mass_m3) {
      failures.push({ step: sample.step, stage: "mass",
        message: `conserved mass drifted by ${sample.conservedMass_m3 - initial.conservedMass_m3} m3` });
    }
    if (sample.topologyEpoch !== previous.topologyEpoch
      && Math.abs(sample.handoffDrift_m3) > tolerance.mass_m3) {
      failures.push({ step: sample.step, stage: "handoff",
        message: `topology handoff drifted by ${sample.handoffDrift_m3} m3` });
    }
    if (sample.massComponents > initial.massComponents) {
      failures.push({ step: sample.step, stage: "mass",
        message: `rho surface split into ${sample.massComponents} components` });
    }
    const reconstructionDelta = sample.reconstructedVolume_m3 - sample.massVisibleVolume_m3;
    if (Math.abs(reconstructionDelta) > tolerance.volume_m3
      || sample.reconstructedComponents > sample.massComponents) {
      failures.push({ step: sample.step, stage: "reconstruction",
        message: `rho->phi changed volume by ${reconstructionDelta} m3 and components ${sample.massComponents}->${sample.reconstructedComponents}` });
    }
    if (sample.renderedVolume_m3 !== undefined) {
      const rendererDelta = sample.renderedVolume_m3 - sample.reconstructedVolume_m3;
      const renderedComponents = sample.renderedComponents ?? sample.reconstructedComponents;
      if (Math.abs(rendererDelta) > tolerance.volume_m3
        || renderedComponents > sample.reconstructedComponents) {
        failures.push({ step: sample.step, stage: "renderer",
          message: `renderer changed volume by ${rendererDelta} m3 and components ${sample.reconstructedComponents}->${renderedComponents}` });
      }
    }
    previous = sample;
  }
  return Object.freeze(failures);
}
