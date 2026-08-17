import {
  createSparseCM12SRR1RuntimePlan,
  type SparseCM12SRR1RuntimePlan,
} from "./sparse-cm12-srr1-runtime-adapter";

export interface SparseCM12SRR1PairedRuntimePlans {
  readonly temporal: SparseCM12SRR1RuntimePlan;
  readonly immutableFullOracle: SparseCM12SRR1RuntimePlan;
}

export interface SparseCM12SRR1PhysicalHashes {
  readonly densitySha256: string;
  readonly velocitySha256: string;
  readonly pressureSha256: string;
  readonly divergenceSha256: string;
}

export interface SparseCM12SRR1PairedCheckpoint {
  readonly step: number;
  readonly temporal: SparseCM12SRR1PhysicalHashes;
  readonly immutableFullOracle: SparseCM12SRR1PhysicalHashes;
}

/** Both adapters have identical ABI/capacity; only construction-static mode differs. */
export function createSparseCM12SRR1PairedRuntimePlans(options: {
  readonly baseWords: number;
  readonly tileCapacity: number;
}): SparseCM12SRR1PairedRuntimePlans {
  const temporal = createSparseCM12SRR1RuntimePlan({ ...options,
    constructionMode: "temporal" });
  const immutableFullOracle = createSparseCM12SRR1RuntimePlan({ ...options,
    constructionMode: "immutable-full-oracle" });
  const sameABI = temporal.authorityLayout.totalWords
      === immutableFullOracle.authorityLayout.totalWords
    && temporal.ingressLayout.totalWords === immutableFullOracle.ingressLayout.totalWords
    && temporal.pipelines.map(({ entryPoint }) => entryPoint).join("\0")
      === immutableFullOracle.pipelines.map(({ entryPoint }) => entryPoint).join("\0");
  if (!sameABI) throw new Error("SRR1 paired oracle ABI diverged");
  return Object.freeze({ temporal, immutableFullOracle });
}

/** No tolerances, active-slot filtering, or diagnostic canonicalization. */
export function assertSparseCM12SRR1PairedPhysicalHashes(
  checkpoint: SparseCM12SRR1PairedCheckpoint,
): void {
  for (const field of ["densitySha256", "velocitySha256", "pressureSha256",
    "divergenceSha256"] as const) {
    if (checkpoint.temporal[field] !== checkpoint.immutableFullOracle[field]) {
      throw new Error(`SRR1 paired step ${checkpoint.step} ${field} mismatch: `
        + `${checkpoint.temporal[field]} != ${checkpoint.immutableFullOracle[field]}`);
    }
  }
}

export function assertSparseCM12SRR1PairedTrajectory(
  checkpoints: readonly SparseCM12SRR1PairedCheckpoint[],
  requiredSteps: readonly number[],
): void {
  const byStep = new Map(checkpoints.map((checkpoint) => [checkpoint.step, checkpoint]));
  for (const step of requiredSteps) {
    const checkpoint = byStep.get(step);
    if (!checkpoint) throw new Error(`SRR1 paired checkpoint ${step} missing`);
    assertSparseCM12SRR1PairedPhysicalHashes(checkpoint);
  }
}
