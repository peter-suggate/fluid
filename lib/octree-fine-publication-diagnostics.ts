export type FinePublicationGateState = "ready" | "failed" | "waiting" | "not-required";

export interface FinePublicationDiagnosticInput {
  readonly generation?: number;
  readonly topologyFlags?: number;
  readonly downstreamReason?: number;
  readonly published?: boolean;
  readonly rolledBack?: boolean;
  readonly redistanceCommitted?: boolean;
  readonly redistanceUnresolved?: number;
  readonly volumeFlags?: number;
  readonly transportCommitted?: boolean;
  readonly transportOutside?: number;
  readonly transportUnavailable?: number;
  readonly transportFaceBandUnavailable?: number;
  readonly faceBandFlags?: number;
  readonly faceBandTransitionFlags?: number;
  readonly faceBandPowerFlags?: number;
  readonly faceBandTransientFlags?: number;
  readonly faceBandPointFlags?: number;
}

export interface FinePublicationGateDiagnostic {
  readonly id: "transport" | "topology" | "redistance" | "volume" | "section5" | "publication";
  readonly label: string;
  readonly state: FinePublicationGateState;
  readonly detail: string;
}

const PUBLISHED = 0x8000_0000;

const labels = (bits: number, definitions: readonly (readonly [number, string])[]) => {
  const names = definitions.filter(([bit]) => (bits & bit) !== 0).map(([, label]) => label);
  return names.length > 0 ? names.join(", ") : "clear";
};

const TOPOLOGY_ERRORS = [
  [1, "capacity"], [2, "hash"], [4, "coarse phi"], [8, "generation"],
] as const;
const FACE_BAND_ERRORS = [
  [1, "capacity"], [2, "hash"], [4, "source"], [8, "row"], [16, "face"],
  [32, "phi"], [64, "unresolved"], [128, "incomplete"], [256, "outside fine band"],
] as const;
const TRANSITION_ERRORS = [
  [1, "source"], [2, "capacity"], [4, "adjacency"], [8, "descriptor"], [16, "acute grading"],
] as const;
const POWER_ERRORS = [
  [1, "source"], [2, "capacity"], [4, "missing row"], [8, "face"],
  [16, "normal"], [32, "nonfinite"], [64, "incomplete"],
] as const;
const POINT_ERRORS = [
  [1, "source"], [2, "capacity"], [4, "face"], [8, "sample"], [16, "normal"],
  [32, "nonfinite"], [64, "singular"], [128, "conditioning"],
] as const;

/**
 * Names the paper-order authority gates from the already-read GPU controls.
 * This is presentation-only: no stage result is fed back into scheduling.
 */
export function finePublicationGateDiagnostics(
  input: FinePublicationDiagnosticInput,
): readonly FinePublicationGateDiagnostic[] {
  const reason = input.downstreamReason ?? 0;
  const topologyErrors = (input.topologyFlags ?? 0) & 0x0f;
  const bootstrap = (input.generation ?? 0) <= 2 && reason === 0 && input.published === true;
  const section5Flags = (input.faceBandFlags ?? 0) | (input.faceBandTransitionFlags ?? 0)
    | (input.faceBandPowerFlags ?? 0) | (input.faceBandTransientFlags ?? 0)
    | (input.faceBandPointFlags ?? 0);
  const section5Failed = section5Flags !== 0 || (input.transportFaceBandUnavailable ?? 0) > 0;

  return [
    {
      id: "section5", label: "Section 5 velocity band",
      state: section5Failed ? "failed" : input.transportCommitted ? "ready" : bootstrap ? "not-required" : "waiting",
      detail: section5Failed
        ? `band ${labels(input.faceBandFlags ?? 0, FACE_BAND_ERRORS)}; transition ${labels(input.faceBandTransitionFlags ?? 0, TRANSITION_ERRORS)}; power ${labels(input.faceBandPowerFlags ?? 0, POWER_ERRORS)}; transient ${labels(input.faceBandTransientFlags ?? 0, POINT_ERRORS)}; point ${labels(input.faceBandPointFlags ?? 0, POINT_ERRORS)}`
        : "regular faces, transition tetrahedra, and point field clear",
    },
    {
      id: "transport", label: "Advect current fine phi",
      state: input.transportCommitted ? "ready" : (reason & 8) !== 0 ? "failed" : bootstrap ? "not-required" : "waiting",
      detail: input.transportCommitted ? "m-segment trace committed"
        : `${input.transportOutside ?? 0} outside band, ${input.transportUnavailable ?? 0} velocity unavailable`,
    },
    {
      id: "topology", label: "Build next fine SPGrid",
      state: topologyErrors === 0 && (reason & 1) === 0 ? "ready" : "failed",
      detail: labels(topologyErrors, TOPOLOGY_ERRORS),
    },
    {
      id: "redistance", label: "Fast march fine phi",
      state: input.redistanceCommitted ? "ready" : (reason & 2) !== 0 ? "failed" : "waiting",
      detail: input.redistanceCommitted ? "physical band committed"
        : `${input.redistanceUnresolved ?? 0} unresolved samples`,
    },
    {
      id: "volume", label: "Conserve enclosed volume",
      state: input.volumeFlags === PUBLISHED ? "ready" : (reason & 4) !== 0 ? "failed" : "waiting",
      detail: input.volumeFlags === PUBLISHED ? "correction published" : `control 0x${(input.volumeFlags ?? 0).toString(16)}`,
    },
    {
      id: "publication", label: "Publish fine generation",
      state: input.published && !input.rolledBack && reason === 0 ? "ready"
        : input.rolledBack || reason !== 0 ? "failed" : "waiting",
      detail: reason === 0 ? (input.rolledBack ? "rejected generation retained" : "clean generation")
        : `rejected by ${labels(reason, [[1, "topology"], [2, "redistance"], [4, "volume"], [8, "transport"]])}`,
    },
  ];
}
