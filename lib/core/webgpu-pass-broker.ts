/**
 * Diagnostic only: end the open pass whenever `compute()` asks for a *different*
 * label, so a labelled pass brackets exactly the dispatches recorded under that
 * label.
 *
 * Without it, a `compute({label})` call whose pass is already open silently
 * DROPS the label. A per-pass timestamp report then names each pass after the
 * first `compute()` since the last fence while charging it everything encoded
 * until the next one. Measured on the mini dam lane 2026-07-28: the five-
 * dispatch workset publication, worth 0.19 ms, reported 3.670 ms on
 * `Workset scatter rows` because that pass stayed open across the downstream
 * stages. With this on it reports 0.011 ms, and the number follows the kernel
 * when the two scatter dispatches are swapped instead of following the slot.
 *
 * Splitting a pass is semantically neutral -- WebGPU already orders storage
 * accesses within a pass, and a pass boundary is a strictly stronger barrier --
 * so this only ever costs pass launches. It measured 39.07 ms/advance against
 * 39.09 ms unisolated on the mini lane, but it does change the command stream,
 * so it stays off unless asked for.
 */
export function passBrokerLabelIsolationRequested(
  environment: Record<string, string | undefined> | undefined
    // The broker encodes in the browser as well as under Dawn, where there is
    // no `process`.
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_GPU_ISOLATE_PASS_LABELS === "1";
}

/** Optional diagnostic scope. An empty list preserves full label isolation;
 * prefixes isolate only transitions into, within, and out of matching stages. */
export function passBrokerLabelIsolationPrefixes(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): readonly string[] {
  return (environment?.FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES ?? "")
    .split(",").map((value) => value.trim()).filter((value) => value.length > 0);
}

/** Dawn/Metal currently drops application encoder labels containing non-ASCII
 * punctuation from xctrace, turning a named pass into `Compute Command N`.
 * Keep the source-facing label expressive, but emit a stable printable form. */
export function xctraceSafeComputeLabel(label: string): string {
  return label
    .replaceAll("·", "-")
    .replaceAll("→", "to")
    .replaceAll("←", "from")
    .replaceAll("×", "x")
    .replace(/[^\x20-\x7e]/g, "?");
}

/** The name a report must use for a pass whose `compute()` carried no label. */
export const UNLABELLED_PASS = "(unlabelled)";

export interface PassBrokerBoundaryAuditBucket {
  /** Calls which requested this boundary, including idempotent fences. */
  readonly requests: number;
  /** Requests which actually closed an open compute pass. */
  readonly passClosures: number;
  /** Copy/clear commands emitted immediately after the boundary. */
  readonly copyCommands: number;
  readonly clearCommands: number;
  /** Exact command bytes when the caller supplied a numeric size. */
  readonly commandBytes: number;
}

type MutableBoundaryBucket = {
  requests: number; passClosures: number; copyCommands: number;
  clearCommands: number; commandBytes: number;
};

const emptyBoundaryBucket = (): MutableBoundaryBucket =>
  ({ requests: 0, passClosures: 0, copyCommands: 0, clearCommands: 0, commandBytes: 0 });

/**
 * Process-wide boundary census, merged by EVERY broker.
 *
 * A broker owns exactly one command encoder, and roughly thirty call sites
 * construct one per encoder and drop it at `finish()`. A per-instance
 * `boundaryAudit` is therefore unreadable by any harness: by the time a report
 * exists, every broker that produced the frame's boundaries is garbage. These
 * module-level totals are merged inside the single boundary funnel, so they
 * cover present and future call sites without any of them being touched.
 *
 * Bounded by the distinct fence-reason vocabulary (tens of entries), not by
 * encoders or frames, and written only where a pass boundary already happens.
 */
const processBoundaries = new Map<string, MutableBoundaryBucket>();
/** Process-wide equivalent of `PassBroker.labelAttributionAudit`: which stage
 * labels are ALREADY sharing a pass with another stage, which is the set of
 * merges the frame has already made. Deduplicated, so it is bounded by the
 * source's distinct label vocabulary rather than by frames or encoders. */
const processAbsorbed = new Map<string, Set<string>>();
let processBrokers = 0;
let processLabelIsolatedBrokers = 0;
const processLabelIsolationPrefixes = new Set<string>();
let processBoundaryResets = 0;

/** Serializable form of the process-wide census, for NDJSON result records. */
export interface PassBrokerBoundaryAuditSnapshot {
  readonly schemaVersion: 1;
  /** Brokers constructed since the last reset; one per command encoder. */
  readonly brokers: number;
  /** Of those, how many were sampling `FLUID_GPU_ISOLATE_PASS_LABELS`. */
  readonly labelIsolatedBrokers: number;
  /**
   * True when ANY contributing broker injected a synthetic
   * `"pass label isolation"` fence per label change. Such a run is a
   * diagnostic command stream: the boundary table carries a large fake bucket
   * and every requests/passClosures ratio is shifted. Recorded from the
   * brokers themselves rather than re-read from the environment, so a snapshot
   * cannot disagree with the stream it describes.
   */
  readonly labelIsolated: boolean;
  /** Union of the isolation prefixes those brokers scoped themselves to. */
  readonly labelIsolationPrefixes?: readonly string[];
  /**
   * How many times the totals were reset in this process. Zero means the
   * census still includes construction and cold bootstrap encoding
   * (`encodeColdTopologySignatureBaseline`, `encodeAnalyticBootstrap`), which
   * inflates any per-advance normalization of it.
   */
  readonly resets: number;
  readonly requests: number;
  readonly passClosures: number;
  readonly copyCommands: number;
  readonly clearCommands: number;
  readonly commandBytes: number;
  /** Distinct (encoder label, dropped label) pairs: passes already merged. */
  readonly absorbedLabelPairs: number;
  /** Encoder labels which name a composite bucket rather than one stage. */
  readonly compositeEncoderLabels: number;
  /** Ranked by real closures, then command traffic, then bytes, then name. */
  readonly byReason: Readonly<Record<string, PassBrokerBoundaryAuditBucket>>;
}

/** The one ranking rule: real closures first, then command traffic. Exported so
 * a report orders its JSON exactly as `formatPassBrokerBoundaryAudit` prints. */
export function rankPassBrokerBoundaryAudit<Bucket extends PassBrokerBoundaryAuditBucket>(
  audit: ReadonlyMap<string, Bucket>,
): [string, Bucket][] {
  return [...audit.entries()].sort((left, right) =>
    right[1].passClosures - left[1].passClosures
    || (right[1].copyCommands + right[1].clearCommands)
      - (left[1].copyCommands + left[1].clearCommands)
    || right[1].commandBytes - left[1].commandBytes
    || left[0].localeCompare(right[0]));
}

/** Every boundary every broker in this process has requested since the last
 * reset. Copied on read so a held snapshot cannot drift as encoding continues. */
export function passBrokerBoundaryAuditTotals(): ReadonlyMap<string, PassBrokerBoundaryAuditBucket> {
  return new Map([...processBoundaries].map(([reason, bucket]) => [reason, { ...bucket }]));
}

/** The same census as a plain, ranked, JSON-serializable record. */
export function passBrokerBoundaryAuditSnapshot(): PassBrokerBoundaryAuditSnapshot {
  const ranked = rankPassBrokerBoundaryAudit(processBoundaries);
  const totals = emptyBoundaryBucket();
  for (const [, bucket] of ranked) {
    totals.requests += bucket.requests;
    totals.passClosures += bucket.passClosures;
    totals.copyCommands += bucket.copyCommands;
    totals.clearCommands += bucket.clearCommands;
    totals.commandBytes += bucket.commandBytes;
  }
  let absorbedLabelPairs = 0;
  for (const swallowed of processAbsorbed.values()) absorbedLabelPairs += swallowed.size;
  return {
    schemaVersion: 1,
    brokers: processBrokers,
    labelIsolatedBrokers: processLabelIsolatedBrokers,
    labelIsolated: processLabelIsolatedBrokers > 0,
    ...(processLabelIsolationPrefixes.size > 0
      ? { labelIsolationPrefixes: [...processLabelIsolationPrefixes].sort() } : {}),
    resets: processBoundaryResets,
    ...totals,
    absorbedLabelPairs,
    compositeEncoderLabels: processAbsorbed.size,
    byReason: Object.fromEntries(ranked.map(([reason, bucket]) => [reason, { ...bucket }])),
  };
}

/**
 * Start a new measurement window. A harness calls this once the solver is warm,
 * exactly where it resets its command audit, so construction and cold bootstrap
 * encoding do not inflate a per-advance number.
 */
export function resetPassBrokerBoundaryAuditTotals(): void {
  processBoundaries.clear();
  processAbsorbed.clear();
  processBrokers = 0;
  processLabelIsolatedBrokers = 0;
  processLabelIsolationPrefixes.clear();
  processBoundaryResets += 1;
}

/**
 * Lazily owns the compute pass for one GPU command encoder.
 *
 * A broker may be threaded through several encoding helpers. Consecutive
 * compute stages then share one pass until a command that cannot be recorded
 * inside a compute pass, or an explicit semantic fence, closes it.
 */
export class PassBroker {
  private openComputePass?: GPUComputePassEncoder;
  private openComputePassLabel?: string;
  private openedComputePassCount = 0;
  private latestFence?: string;
  /**
   * Every label that `compute()` was asked for while a pass opened under a
   * DIFFERENT label was already open, keyed by that opening label.
   *
   * Metal names an encoder once, when it begins, so a later label cannot reach
   * the trace: the encoder keeps the opening name and Instruments charges it
   * everything encoded until the pass ends. That is not a defect to be fixed by
   * fencing -- production deliberately shares passes -- it is a fact about the
   * measurement that has to be *reported*, because a per-encoder timing report
   * otherwise prints a composite bucket under a single stage's name.
   *
   * Deduplicated, so a stage encoded 165 times costs one entry; bounded by the
   * distinct label count of one command encoder, which is a few hundred.
   */
  private readonly absorbed = new Map<string, Set<string>>();
  /** Structural census of every requested pass boundary. This deliberately
   * records idempotent requests as well as real closures: the former identify
   * callers which can be batched, while the latter are the serial spine. */
  private readonly boundaries = new Map<string, MutableBoundaryBucket>();
  /** Sampled per broker so a test can drive it; production reads the env once
   * per command encoder, which is not on any hot path. */
  private readonly isolateLabels: boolean;
  private readonly isolateLabelPrefixes: readonly string[];

  constructor(private readonly encoder: GPUCommandEncoder, options?: {
    isolateLabels?: boolean;
    isolateLabelPrefixes?: readonly string[];
  }) {
    this.isolateLabels = options?.isolateLabels ?? passBrokerLabelIsolationRequested();
    this.isolateLabelPrefixes = (options?.isolateLabelPrefixes
      ?? passBrokerLabelIsolationPrefixes()).map(xctraceSafeComputeLabel);
    // Census the isolation state here rather than re-reading the environment at
    // report time: a report must describe the stream that was actually encoded.
    processBrokers += 1;
    if (this.isolateLabels) {
      processLabelIsolatedBrokers += 1;
      for (const prefix of this.isolateLabelPrefixes) processLabelIsolationPrefixes.add(prefix);
    }
  }

  /** Return the open compute pass, creating it only when necessary. */
  compute(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    const requestedLabel = descriptor?.label;
    const nextLabel = requestedLabel === undefined
      ? undefined : xctraceSafeComputeLabel(requestedLabel);
    const emittedDescriptor = requestedLabel === nextLabel
      ? descriptor
      : { ...descriptor, label: nextLabel };
    const scopedIsolation = this.isolateLabelPrefixes.length === 0
      || this.isolateLabelPrefixes.some((prefix) =>
        this.openComputePassLabel?.startsWith(prefix) || nextLabel?.startsWith(prefix));
    // A targeted pass must also close before an unlabelled request; otherwise
    // the unlabelled dispatch would silently become part of the target bucket.
    const labelChanged = nextLabel === undefined
      ? this.isolateLabelPrefixes.some((prefix) => this.openComputePassLabel?.startsWith(prefix))
      : nextLabel !== this.openComputePassLabel;
    if (this.isolateLabels && this.openComputePass && labelChanged && scopedIsolation) {
      this.fence("pass label isolation");
    }
    if (!this.openComputePass) {
      this.openComputePass = this.encoder.beginComputePass(emittedDescriptor);
      this.openComputePassLabel = nextLabel;
      this.openedComputePassCount += 1;
      return this.openComputePass;
    }
    // The pass survived, so `emittedDescriptor.label` never reaches Metal.
    // Record the loss rather than let it be inferred from a timing report.
    if (nextLabel !== this.openComputePassLabel) {
      const owner = this.openComputePassLabel ?? UNLABELLED_PASS;
      const dropped = nextLabel ?? UNLABELLED_PASS;
      for (const map of [this.absorbed, processAbsorbed]) {
        const swallowed = map.get(owner) ?? new Set<string>();
        swallowed.add(dropped);
        map.set(owner, swallowed);
      }
    }
    return this.openComputePass;
  }

  /** End the current pass. `reason` documents semantic boundaries at callers. */
  fence(reason?: string): void {
    const name = reason ?? "explicit fence";
    const buckets = this.boundaryBuckets(name);
    for (const bucket of buckets) bucket.requests += 1;
    const pass = this.openComputePass;
    if (!pass) return;
    for (const bucket of buckets) bucket.passClosures += 1;
    this.openComputePass = undefined;
    this.openComputePassLabel = undefined;
    this.latestFence = reason;
    pass.end();
  }

  clearBuffer(buffer: GPUBuffer, offset?: GPUSize64, size?: GPUSize64): void {
    this.fence("clear buffer");
    const bytes = typeof size === "number" ? size
      : typeof buffer.size === "number" ? buffer.size - Number(offset ?? 0) : 0;
    for (const bucket of this.boundaryBuckets("clear buffer")) {
      bucket.clearCommands += 1;
      bucket.commandBytes += bytes;
    }
    this.encoder.clearBuffer(buffer, offset, size);
  }

  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: GPUSize64,
    destination: GPUBuffer,
    destinationOffset: GPUSize64,
    size: GPUSize64,
  ): void {
    this.fence("copy buffer");
    for (const bucket of this.boundaryBuckets("copy buffer")) {
      bucket.copyCommands += 1;
      if (typeof size === "number") bucket.commandBytes += size;
    }
    this.encoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
  }

  /** Copy GPU-authored command arguments into an INDIRECT-only buffer. */
  updateIndirectBuffer(
    source: GPUBuffer,
    sourceOffset: GPUSize64,
    destination: GPUBuffer,
    destinationOffset: GPUSize64,
    size: GPUSize64,
  ): void {
    // Keep indirect publication distinct in the audit. It is the dominant
    // recurring copy family and therefore the first Bet-3 staging target.
    this.fence("stage indirect args");
    for (const bucket of this.boundaryBuckets("stage indirect args")) {
      bucket.copyCommands += 1;
      if (typeof size === "number") bucket.commandBytes += size;
    }
    this.encoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
  }

  /**
   * THE boundary funnel. Every write to either census goes through here, so a
   * new boundary kind cannot reach the command stream without being counted,
   * and this broker's own audit can never disagree with the process totals.
   */
  private boundaryBuckets(reason: string): readonly MutableBoundaryBucket[] {
    let bucket = this.boundaries.get(reason);
    if (!bucket) {
      bucket = emptyBoundaryBucket();
      this.boundaries.set(reason, bucket);
    }
    let total = processBoundaries.get(reason);
    if (!total) {
      total = emptyBoundaryBucket();
      processBoundaries.set(reason, total);
    }
    return [bucket, total];
  }

  /** Escape to commands not represented here only after closing compute. */
  commandEncoder(): GPUCommandEncoder {
    this.fence("raw command encoder access");
    return this.encoder;
  }

  /** Finish the underlying encoder without leaving an unterminated pass. */
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer {
    this.fence("finish command encoder");
    return this.encoder.finish(descriptor);
  }

  /** Useful to command audits and focused unit tests; not a synchronization query. */
  get computePassCount(): number { return this.openedComputePassCount; }

  get hasOpenComputePass(): boolean { return this.openComputePass !== undefined; }

  get lastFenceReason(): string | undefined { return this.latestFence; }

  /**
   * Which stage labels this broker dropped, and which pass swallowed them.
   *
   * A key is the label Metal actually recorded for an encoder; its values are
   * the stages that were encoded into that same encoder under a name no trace
   * can see. Any timing attributed to a key that has values is a COMPOSITE
   * bucket -- the key's own dispatches plus every value's -- never that
   * stage's cost.
   */
  get labelAttributionAudit(): ReadonlyMap<string, ReadonlySet<string>> {
    return this.absorbed;
  }

  /** Stages encoded under `label`'s Metal encoder but not named by it. */
  absorbedBy(label: string): readonly string[] {
    return [...this.absorbed.get(xctraceSafeComputeLabel(label)) ?? []];
  }

  /** True when `label` named an encoder that carried nothing else. */
  attributionIsExact(label: string): boolean {
    return !this.absorbed.has(xctraceSafeComputeLabel(label));
  }

  /** Distinct (encoder label, dropped label) pairs this broker produced. */
  get absorbedLabelCount(): number {
    let total = 0;
    for (const swallowed of this.absorbed.values()) total += swallowed.size;
    return total;
  }

  /** Exact per-command-encoder census of boundary causes. */
  get boundaryAudit(): ReadonlyMap<string, PassBrokerBoundaryAuditBucket> {
    return this.boundaries;
  }
}

/**
 * Render a broker's dropped labels as report lines. Kept beside the broker so
 * a harness never has to re-derive the rule that a key with values is a
 * composite bucket rather than a stage.
 */
export function formatPassBrokerLabelAudit(
  audit: ReadonlyMap<string, ReadonlySet<string>>,
  limit = 12,
): readonly string[] {
  const ranked = [...audit.entries()]
    .sort((left, right) => right[1].size - left[1].size);
  if (ranked.length === 0) return [];
  const lines = [`${ranked.length} Metal encoder labels are composite buckets;`
    + " each also carries the stages listed after it:"];
  for (const [owner, swallowed] of ranked.slice(0, limit)) {
    lines.push(`  ${owner} + ${swallowed.size}: ${[...swallowed].join(", ")}`);
  }
  if (ranked.length > limit) lines.push(`  ... and ${ranked.length - limit} more`);
  return lines;
}

/** Rank the serial-spine causes by real pass closures, then command traffic. */
export function formatPassBrokerBoundaryAudit(
  audit: ReadonlyMap<string, PassBrokerBoundaryAuditBucket>,
  limit = 12,
): readonly string[] {
  const ranked = rankPassBrokerBoundaryAudit(audit);
  if (ranked.length === 0) return [];
  const lines = [`${ranked.reduce((sum, [, bucket]) => sum + bucket.passClosures, 0)} compute-pass boundaries by cause:`];
  for (const [reason, bucket] of ranked.slice(0, limit)) {
    lines.push(`  ${reason}: ${bucket.passClosures} closures / ${bucket.requests} requests, `
      + `${bucket.copyCommands} copies, ${bucket.clearCommands} clears, ${bucket.commandBytes} bytes`);
  }
  if (ranked.length > limit) lines.push(`  ... and ${ranked.length - limit} more`);
  return lines;
}
