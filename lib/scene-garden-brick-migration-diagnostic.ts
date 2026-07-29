import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  hookFinding,
  numberPath,
  recordPath,
  runTime,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";

export interface GardenBrickMigrationDiagnosticParameters {
  initialCoreBricks: number;
  evaluateAfter_s: number;
  minimumFinalCoreBricks: number;
  sourceFluidVoxelsAtEnd: number;
  sourceCoreResidencyAtEnd: boolean;
}

export function evaluateGardenBrickMigrationDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: GardenBrickMigrationDiagnosticParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const sparse = recordPath(diagnostics, "sparse") ?? diagnostics;
    const initial = recordPath(sparse, "initialFluidBricks")
      ?? recordPath(diagnostics, "initialFluidBrickStats");
    const final = recordPath(sparse, "finalPublication")
      ?? recordPath(diagnostics, "sparseVoxelStats");
    const initialCore = numberPath(initial, "core");
    findings.push(hookFinding({
      id: `${method}.initial-core`, method,
      passed: initialCore === input.parameters.initialCoreBricks,
      message: initialCore === input.parameters.initialCoreBricks
        ? `migration began with ${initialCore} core brick(s)`
        : `migration scene started with ${initialCore ?? "unknown"} core fluid bricks instead of ${input.parameters.initialCoreBricks}`,
      expected: input.parameters.initialCoreBricks, actual: initialCore,
    }));

    const resident = numberPath(final, "fluidBrickResidentCount");
    const capacity = numberPath(final, "fluidBrickCapacity");
    findings.push(hookFinding({
      id: `${method}.resident-capacity`, method,
      passed: resident !== undefined && capacity !== undefined && resident < capacity,
      message: resident !== undefined && capacity !== undefined && resident < capacity
        ? `migration resident set remained below its ${capacity}-brick capacity`
        : `migration resident set filled or lacked its ${capacity ?? "unknown"}-brick capacity`,
      expected: { lessThanCapacity: capacity }, actual: resident,
    }));

    const elapsed = runTime(diagnostics);
    if (elapsed === undefined || elapsed < input.parameters.evaluateAfter_s - 1e-9) continue;
    const finalCore = numberPath(final, "fluidBrickCoreCount");
    const sourceVoxels = numberPath(final, "sourceBrickFluidVoxelCount");
    const sourceResidency = recordPath(final)?.sourceBrickResidency;
    findings.push(hookFinding({
      id: `${method}.final-core`, method,
      passed: finalCore !== undefined && finalCore >= input.parameters.minimumFinalCoreBricks,
      message: finalCore !== undefined && finalCore >= input.parameters.minimumFinalCoreBricks
        ? `migration reached ${finalCore} core bricks`
        : `migration did not reach ${input.parameters.minimumFinalCoreBricks} core bricks`,
      expected: { minimum: input.parameters.minimumFinalCoreBricks }, actual: finalCore,
    }));
    findings.push(hookFinding({
      id: `${method}.source-evacuated`, method,
      passed: sourceVoxels === input.parameters.sourceFluidVoxelsAtEnd,
      message: sourceVoxels === input.parameters.sourceFluidVoxelsAtEnd
        ? "migration evacuated the authored source brick"
        : `migration left ${sourceVoxels ?? "unknown"} liquid voxels in its original brick`,
      expected: input.parameters.sourceFluidVoxelsAtEnd, actual: sourceVoxels,
    }));
    const sourceIsCore = sourceResidency === "core";
    findings.push(hookFinding({
      id: `${method}.source-residency`, method,
      passed: sourceIsCore === input.parameters.sourceCoreResidencyAtEnd,
      message: sourceIsCore === input.parameters.sourceCoreResidencyAtEnd
        ? "migration source residency matches the declared terminal state"
        : `migration source core residency ${sourceIsCore} differs from ${input.parameters.sourceCoreResidencyAtEnd}`,
      expected: input.parameters.sourceCoreResidencyAtEnd, actual: sourceIsCore,
    }));
  }
  return findings;
}
