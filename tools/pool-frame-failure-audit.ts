import { appendFile, writeFile } from "node:fs/promises";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Read-only receipts after every complete frame, including unsampled frames. */
export function createPoolFrameFailureAudit(solver: WebGPUAdaptiveMassSolver, output: string) {
  const rows: Record<string, unknown>[] = [];
  return {
    async capture(step: number, time: number) {
      const [stats, activity, frame, publication, effects, scalarMask, extension] = await Promise.all([
        solver.readStats(), solver.readGPUActivityPolicy(), solver.readFrameControlQA(),
        solver.readFramePlanPresentationHeaderQA(), solver.readCandidateEffectsTransactionQA(),
        solver.readFinalScalarMaskHeaderQA(), solver.readVelocityExtensionHeaderQA(),
      ]);
      const pressure = Object.fromEntries(Object.entries(stats).filter(([key]) =>
        key.startsWith("pressure") || key.startsWith("adaptivePressure")
        || key.startsWith("topologyGeneration") || ["encodedSteps", "completedTime_s",
          "submittedTime_s", "fluidBrickGeneration", "maxDivergenceAfter_s"].includes(key)));
      const rejected = activity.bricks.filter(b => b.candidateStatus === 2
        || b.transferStatus === 2 || b.faceTransferStatus === 2).map(b => ({
        leafId: b.leafId, coordinate: b.coordinate, active: b.active,
        acceptedResolution: b.acceptedResolution, plannedResolution: b.plannedResolution,
        candidateResolution: b.candidateResolution, candidateStatus: b.candidateStatus,
        transferStatus: b.transferStatus, faceTransferStatus: b.faceTransferStatus,
        planReasons: b.planReasons, representabilityFailure: b.representabilityFailure,
        massError: b.transferMassErrorFineCells, momentumError: b.transferMomentumErrorFineCells,
        fluxError: b.maximumAbsoluteTransferFluxErrorFineAreas,
      }));
      const row = { step, time, pressure, frame, publication, effects: effects?.tfx,
        scalarMask, extension, topology: {
          acceptedSteps: activity.acceptedSteps,
          generation: activity.acceptedTopologyGeneration, faultFlags: activity.faultFlags,
          commitFailed: activity.commitFailed, prepared: activity.preparedBrickCount,
          committed: activity.committedBrickCount, allocator: activity.topologyPageAllocator,
          rejected,
        } };
      rows.push(row);
      if (step === 0) await writeFile(`${output}/frame-audit.jsonl`, "");
      await appendFile(`${output}/frame-audit.jsonl`, `${JSON.stringify(row)}\n`);
      const failure = frame.fault || publication.faultCode || scalarMask.fault
        || extension.faultCount || activity.commitFailed || (effects?.tfx.fault ?? 0)
        || (step > 0 && stats.pressureIterationCapReached);
      if (failure || activity.faultFlags) console.log(JSON.stringify({ auditEvent: row }));
    },
    async finish() {
      await writeFile(`${output}/frame-audit.json`, JSON.stringify(rows, null, 2));
    },
  };
}
