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
  readonly transportStructuredAuthorityUnavailable?: number;
}

export interface FinePublicationGateDiagnostic {
  readonly id: "transport" | "topology" | "redistance" | "volume" | "publication";
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
  return [
    {
      id: "transport", label: "Advect current fine phi",
      state: input.transportCommitted ? "ready"
        : (reason & 8) !== 0 || (input.transportStructuredAuthorityUnavailable ?? 0) > 0
          ? "failed" : bootstrap ? "not-required" : "waiting",
      detail: input.transportCommitted ? "m-segment trace committed"
        : `${input.transportOutside ?? 0} outside band, ${input.transportUnavailable ?? 0} velocity unavailable, ${input.transportStructuredAuthorityUnavailable ?? 0} structured authority unavailable`,
    },
    {
      id: "topology", label: "Build next fine SPGrid",
      state: topologyErrors === 0 && (reason & 1) === 0 ? "ready" : "failed",
      detail: labels(topologyErrors, TOPOLOGY_ERRORS),
    },
    {
      id: "redistance", label: "JFA-CPT redistance fine phi",
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
