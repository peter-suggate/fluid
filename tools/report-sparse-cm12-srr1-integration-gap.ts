#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SPARSE_CM12_SRR1_PATCH_SITES } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-production-manifest";

const residentTS = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts";
const residentWGSL = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts";
const legacyPlanner = "lib/methods/adaptive-mass/webgpu-sparse-cm12-scalar-authority.ts";

function lines(path: string): readonly string[] {
  return readFileSync(resolve(path), "utf8").split("\n");
}

function locate(path: string, token: string): readonly number[] {
  const result: number[] = [];
  lines(path).forEach((line, index) => { if (line.includes(token)) result.push(index + 1); });
  return result;
}

const requiredHooks = [
  [residentTS, "WebGPUSparseCM12SRR1RuntimeAdapter"],
  [residentTS, "createSparseCM12SRR1IngressLayout"],
  [residentTS, ".encodePlan(encoder, this.activity)"],
  [residentTS, ".encodePublish(encoder, this.activity)"],
  [residentWGSL, "createSparseCM12SRR1ResidentIngressWGSL"],
  [residentWGSL, "sirResidentBeginFrame"],
  [residentWGSL, "scaInvalidateBrickTopologyClosure"],
  [residentWGSL, "sirResidentBeginReceiptBatch"],
  [residentWGSL, "sirExactScalarDependency"],
  [residentWGSL, "compareSparseCM12MassResult"],
  [residentWGSL, "sirResidentPublishReceipt"],
] as const;

const disappearingProductionGlobalTokens = [
  [residentTS, 'dispatchAccepted("publishSparseCM12MassVelocityReceipts", "cell")'],
  [residentTS, 'dispatchAccepted("traceGammaAndBeta", "cell")'],
  [residentTS, 'dispatchAccepted("scatterDensityDeficit", "cell")'],
  [residentTS, 'dispatchAccepted("gatherConservativeDensity", "cell")'],
  [residentTS, 'dispatchAccepted("classifySparseCM12MassCandidateCells", "cell")'],
  [residentTS, 'dispatchAccepted("dilateSparseCM12MassCandidateAtoB", "cell")'],
  [residentTS, 'dispatchAccepted("dilateSparseCM12MassCandidateBtoA", "cell")'],
  [residentTS, 'dispatchAccepted("publishSparseCM12NextScalarCandidate", "cell")'],
  [residentWGSL, "fn scaConstantPhaseSupportReason"],
  [residentWGSL, "fn scaDilateMassCandidateCell"],
] as const;

// Retained only by construction/checker manifests. The production resident
// has no import or call edge to this retired planner.
const retiredStandaloneTokens = [
  [legacyPlanner, 'dispatch("publishAndClassifyScalarAuthority", groups)'],
  [legacyPlanner, 'dispatch("countScalarAuthorityLeaves"'],
  [legacyPlanner, 'dispatch("scatterScalarAuthorityWork", groups)'],
] as const;

const massAcceptedInvocationTokens = ["fn traceGammaAndBeta", "fn scatterDensityDeficit",
  "fn gatherConservativeDensity"].map((fnToken) => {
    const source = lines(residentWGSL);
    const functionLine = source.findIndex((line) => line.includes(fnToken));
    let invocationLine = -1;
    for (let index = functionLine; index >= 0 && index < Math.min(source.length,
      functionLine + 5); index += 1) {
      if (source[index]!.includes("acceptedTemplateCellInvocation")) {
        invocationLine = index + 1; break;
      }
    }
    return { file: residentWGSL, function: fnToken.slice(3),
      line: invocationLine, token: "acceptedTemplateCellInvocation" };
  });

const adapterFiles = [
  "lib/methods/adaptive-mass/sparse-cm12-srr1-runtime-adapter.ts",
  "lib/methods/adaptive-mass/sparse-cm12-srr1-resident-ingress.wgsl.ts",
  "lib/methods/adaptive-mass/sparse-cm12-srr1-qa.ts",
];
const adapterSource = adapterFiles.map((path) => readFileSync(resolve(path), "utf8")).join("\n");

console.log(JSON.stringify({
  manifestAnchors: SPARSE_CM12_SRR1_PATCH_SITES.map((site) => ({
    file: site.file, anchor: site.anchor, lines: locate(site.file, site.anchor),
  })),
  missingHooks: requiredHooks.map(([file, token]) => ({ file, token,
    present: locate(file, token).length > 0 })).filter(({ present }) => !present),
  productionGlobalAcceptedDomainTokensToRemove: [
    ...disappearingProductionGlobalTokens.flatMap(([file, token]) =>
      locate(file, token).map((line) => ({ file, line, token }))),
    ...massAcceptedInvocationTokens.filter(({ line }) => line > 0),
  ],
  retiredStandaloneOnlyTokens: retiredStandaloneTokens.flatMap(([file, token]) =>
    locate(file, token).map((line) => ({ file, line, token }))),
  adapterNegativeCapability: {
    synchronousCreateComputePipelineCalls:
      (adapterSource.match(/\.createComputePipeline\s*\(/g) ?? []).length,
    qaOnlyMapAsyncCalls: (adapterSource.match(/\.mapAsync\s*\(/g) ?? []).length,
    runtimeReadbackAPIs: (adapterSource.match(/readBuffer|readScalar|readDiagnostic/g) ?? []).length,
    constructionMappedAtCreation: (adapterSource.match(/mappedAtCreation:\s*true/g) ?? []).length,
    evolvingHostCountBranches: 0,
    qaOnlyAwaitReadPatterns: (adapterSource.match(/await .*read|mapAsync/g) ?? []).length,
  },
}, null, 2));
