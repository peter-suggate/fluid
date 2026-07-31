import {
  packSvoNodeMipPageKey,
  svoNodeMipPageKey,
  type SvoNodeMipPageKey,
  type SvoNodeMipPyramidPlan,
} from "./svo-node-mip-pyramid";
import {
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  type SvoSceneLights,
} from "./svo-light-abi";
import { hashSvoStaticPublication } from "./svo-static-publication-cache";

/**
 * A page/light channel is a conservative certificate, not a filtered shadow.
 * Two bitplanes encode 32 channels in eight bytes per shared node-mip slot.
 */
export const SVO_STATIC_SHADOW_FIELD_LAYOUT = Object.freeze({
  version: 1,
  maximumLights: SVO_LIGHT_MAXIMUM_RECORDS,
  visibleMaskWord: 0,
  occludedMaskWord: 1,
  wordsPerPage: 2,
  bytesPerPage: 2 * Uint32Array.BYTES_PER_ELEMENT,
} as const);

export type SvoStaticShadowCertificate = "unknown" | "visible" | "occluded" | "mixed";

export interface SvoStaticShadowLightChannel {
  /** Index in the renderer's fixed-size SVO light ABI. */
  channel: number;
  lightId: number;
  lightRevision: number;
}

export interface SvoStaticShadowPagePlan {
  key: SvoNodeMipPageKey;
  keyString: string;
  /** Exactly the shared node-mip physical slot; no second page table is needed. */
  slot: number;
}

export interface SvoStaticShadowFieldPlan {
  cacheKey: string;
  sourceGeneration: number;
  lightRevision: number;
  lightStaticRevision: string;
  topologyHash: string;
  lightHash: string;
  pages: readonly SvoStaticShadowPagePlan[];
  lightChannels: readonly SvoStaticShadowLightChannel[];
  requiredLightMask: number;
  atlasCapacity: number;
  residentPageCount: number;
  /** Resident useful bytes; atlas allocation can be larger when its shape has spare slots. */
  residentPayloadBytes: number;
  allocatedBytes: number;
  /** Double buffering is optional, but this is the safe peak while replacing a visible field. */
  maximumReplacementBytes: number;
}

export interface SvoStaticShadowPageCertificates {
  page: SvoNodeMipPageKey | string;
  certificates: readonly {
    lightId: number;
    certificate: Exclude<SvoStaticShadowCertificate, "unknown">;
  }[];
}

export type SvoStaticShadowRenderDecision =
  | {
    certificate: "visible";
    staticAction: "skip-static";
    dynamicAction: "trace-current-frame-overlay";
  }
  | {
    certificate: "occluded";
    staticAction: "reject-light";
    dynamicAction: "skip-dynamic";
  }
  | {
    certificate: "mixed" | "unknown";
    staticAction: "trace-static-current-frame";
    dynamicAction: "trace-current-frame-overlay";
  };

export interface SvoStaticShadowFieldSnapshot {
  plan: SvoStaticShadowFieldPlan;
  /** Slot-major `[visibleMask, occludedMask]`; safe to upload as one storage buffer. */
  words: Uint32Array<ArrayBuffer>;
}

export interface SvoStaticShadowFieldTelemetry {
  visibleCacheKey?: string;
  candidateCacheKey?: string;
  residentPages: number;
  readyPages: number;
  dirtyPages: number;
  readyChannels: number;
  dirtyChannels: number;
  allocatedBytes: number;
  fallback: "none" | "exact-current-frame-trace";
}

export interface SvoStaticShadowBuildWork {
  cacheKey: string;
  pageKey: SvoNodeMipPageKey;
  pageKeyString: string;
  slot: number;
  lightId: number;
  lightRevision: number;
  lightChannel: number;
}

export type SvoStaticShadowFieldPublicationDecision =
  | { published: true; reason: "published" | "already-visible"; visible: SvoStaticShadowFieldSnapshot }
  | {
    published: false;
    reason: "no-candidate" | "dirty-pages" | "revision-order" | "revision-collision";
    visible?: SvoStaticShadowFieldSnapshot;
  };

interface OwnedShadowField {
  plan: SvoStaticShadowFieldPlan;
  words: Uint32Array<ArrayBuffer>;
  pageByKey: ReadonlyMap<string, SvoStaticShadowPagePlan>;
  channelByLightId: ReadonlyMap<number, SvoStaticShadowLightChannel>;
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must fit uint32`);
  }
  return value >>> 0;
}

function bit(channel: number): number {
  if (!Number.isInteger(channel) || channel < 0 || channel >= SVO_STATIC_SHADOW_FIELD_LAYOUT.maximumLights) {
    throw new RangeError("SVO static-shadow light channel is outside the fixed ABI");
  }
  return (1 << channel) >>> 0;
}

function requiredMask(count: number): number {
  if (count === 0) return 0;
  return count === 32 ? 0xffff_ffff : (2 ** count - 1) >>> 0;
}

function topologyHash(plan: SvoNodeMipPyramidPlan): string {
  const words = new Uint32Array(6 + plan.pages.length * 5);
  words.set([
    plan.generation,
    plan.residentPageCount,
    plan.atlas.capacity,
    plan.atlas.pages[0],
    plan.atlas.pages[1],
    plan.atlas.pages[2],
  ]);
  let offset = 6;
  for (const page of plan.pages) {
    words.set(packSvoNodeMipPageKey(page.key), offset);
    words[offset + 4] = page.slot;
    offset += 5;
  }
  return hashSvoStaticPublication(words, "svo-static-shadow-topology-v1");
}

/**
 * Plans an immutable static shadow accelerator over the node-mip plan's exact
 * slot topology. Light count and transforms are data, so neither changes the shader.
 */
export function planSvoStaticShadowField(
  nodeMips: SvoNodeMipPyramidPlan,
  lights: Pick<SvoSceneLights, "records" | "packedRecords" | "revision" | "staticRevision">,
): SvoStaticShadowFieldPlan {
  if (!nodeMips.complete) throw new Error("Static-shadow fields require a complete node-mip topology");
  const sourceGeneration = uint32(nodeMips.generation, "SVO static-shadow source generation");
  const lightRevision = uint32(lights.revision, "SVO static-shadow light revision");
  if (lights.records.length > SVO_STATIC_SHADOW_FIELD_LAYOUT.maximumLights) {
    throw new RangeError("SVO static-shadow light table exceeds the fixed ABI");
  }
  if (lights.packedRecords.byteLength !== lights.records.length * SVO_LIGHT_RECORD_STRIDE_BYTES) {
    throw new RangeError("SVO static-shadow packed light table does not match the light ABI");
  }
  const seenLightIds = new Set<number>();
  const lightChannels = lights.records.map((light, channel): SvoStaticShadowLightChannel => {
    const lightId = uint32(light.lightId, "SVO static-shadow light ID");
    if (seenLightIds.has(lightId)) throw new RangeError(`Duplicate SVO static-shadow light ID ${lightId}`);
    seenLightIds.add(lightId);
    if (uint32(light.revision, "SVO static-shadow record revision") !== lightRevision) {
      throw new Error(`SVO static-shadow light ${lightId} does not match publication revision ${lightRevision}`);
    }
    return Object.freeze({ channel, lightId, lightRevision });
  });
  const topology = topologyHash(nodeMips);
  const lightHash = hashSvoStaticPublication(lights.packedRecords, `svo-static-shadow-lights-v1:${lights.staticRevision}`);
  const cacheKey = `svo-static-shadow-v${SVO_STATIC_SHADOW_FIELD_LAYOUT.version}`
    + `:${sourceGeneration}:${lightRevision}:${topology}:${lightHash}`;
  const pages = nodeMips.pages.map(({ key, keyString, slot }) => Object.freeze({ key, keyString, slot }));
  const allocatedBytes = nodeMips.atlas.capacity * SVO_STATIC_SHADOW_FIELD_LAYOUT.bytesPerPage;
  return Object.freeze({
    cacheKey,
    sourceGeneration,
    lightRevision,
    lightStaticRevision: lights.staticRevision,
    topologyHash: topology,
    lightHash,
    pages,
    lightChannels,
    requiredLightMask: requiredMask(lightChannels.length),
    atlasCapacity: nodeMips.atlas.capacity,
    residentPageCount: pages.length,
    residentPayloadBytes: pages.length * SVO_STATIC_SHADOW_FIELD_LAYOUT.bytesPerPage,
    allocatedBytes,
    maximumReplacementBytes: allocatedBytes * 2,
  });
}

function own(plan: SvoStaticShadowFieldPlan): OwnedShadowField {
  return {
    plan,
    words: new Uint32Array(plan.atlasCapacity * SVO_STATIC_SHADOW_FIELD_LAYOUT.wordsPerPage),
    pageByKey: new Map(plan.pages.map((page) => [page.keyString, page])),
    channelByLightId: new Map(plan.lightChannels.map((channel) => [channel.lightId, channel])),
  };
}

function pageKeyString(page: SvoNodeMipPageKey | string): string {
  return typeof page === "string" ? page : svoNodeMipPageKey(page);
}

function masks(owner: OwnedShadowField, slot: number): readonly [number, number] {
  const offset = slot * SVO_STATIC_SHADOW_FIELD_LAYOUT.wordsPerPage;
  return [owner.words[offset], owner.words[offset + 1]];
}

function certificateFromMasks(visibleMask: number, occludedMask: number, channel: number): SvoStaticShadowCertificate {
  const channelBit = bit(channel);
  const visible = (visibleMask & channelBit) !== 0;
  const occluded = (occludedMask & channelBit) !== 0;
  if (visible && occluded) return "mixed";
  if (visible) return "visible";
  if (occluded) return "occluded";
  return "unknown";
}

function decision(certificate: SvoStaticShadowCertificate): SvoStaticShadowRenderDecision {
  if (certificate === "visible") {
    return { certificate, staticAction: "skip-static", dynamicAction: "trace-current-frame-overlay" };
  }
  if (certificate === "occluded") {
    return { certificate, staticAction: "reject-light", dynamicAction: "skip-dynamic" };
  }
  return { certificate, staticAction: "trace-static-current-frame", dynamicAction: "trace-current-frame-overlay" };
}

function popcount(value: number): number {
  let word = value >>> 0;
  let count = 0;
  while (word !== 0) {
    word &= word - 1;
    count += 1;
  }
  return count;
}

function snapshot(owner: OwnedShadowField): SvoStaticShadowFieldSnapshot {
  return { plan: owner.plan, words: owner.words.slice() as Uint32Array<ArrayBuffer> };
}

/**
 * CPU lifecycle and publication contract for a GPU-built field. Certificate
 * writes are expected to follow their payload work in command order. Until a
 * channel is certified, render resolution falls back to the exact current-frame
 * static traversal; an older generation is never treated as current.
 */
export class SvoStaticShadowFieldCache {
  private visible?: OwnedShadowField;
  private candidate?: OwnedShadowField;

  begin(plan: SvoStaticShadowFieldPlan): { reused: boolean; dirtyPages: number; dirtyChannels: number } {
    if (this.visible?.plan.cacheKey === plan.cacheKey) {
      this.candidate = undefined;
      return { reused: true, dirtyPages: 0, dirtyChannels: 0 };
    }
    if (this.candidate?.plan.cacheKey !== plan.cacheKey) this.candidate = own(plan);
    return {
      reused: false,
      dirtyPages: plan.residentPageCount,
      dirtyChannels: plan.residentPageCount * plan.lightChannels.length,
    };
  }

  /** Atomically certifies any subset of a page's light channels. */
  publishPageCertificates(publication: SvoStaticShadowPageCertificates): void {
    if (!this.candidate) throw new Error("No SVO static-shadow candidate exists");
    const key = pageKeyString(publication.page);
    const page = this.candidate.pageByKey.get(key);
    if (!page) throw new Error("SVO static-shadow page is not resident in the candidate topology");
    const offset = page.slot * SVO_STATIC_SHADOW_FIELD_LAYOUT.wordsPerPage;
    let visibleMask = this.candidate.words[offset];
    let occludedMask = this.candidate.words[offset + 1];
    for (const entry of publication.certificates) {
      const channel = this.candidate.channelByLightId.get(entry.lightId);
      if (!channel) throw new Error(`SVO static-shadow light ${entry.lightId} is not in the candidate light table`);
      const existing = certificateFromMasks(visibleMask, occludedMask, channel.channel);
      if (existing !== "unknown" && existing !== entry.certificate) {
        throw new Error(`Conflicting SVO static-shadow certificate for page ${key}, light ${entry.lightId}`);
      }
      const channelBit = bit(channel.channel);
      if (entry.certificate === "visible" || entry.certificate === "mixed") visibleMask |= channelBit;
      if (entry.certificate === "occluded" || entry.certificate === "mixed") occludedMask |= channelBit;
    }
    this.candidate.words[offset] = visibleMask >>> 0;
    this.candidate.words[offset + 1] = occludedMask >>> 0;
  }

  /** Conservative convenience for a page whose channels all require exact traversal. */
  publishPageMixed(page: SvoNodeMipPageKey | string): void {
    if (!this.candidate) throw new Error("No SVO static-shadow candidate exists");
    this.publishPageCertificates({
      page,
      certificates: this.candidate.plan.lightChannels.map(({ lightId }) => ({ lightId, certificate: "mixed" as const })),
    });
  }

  /**
   * Resolves only an exact revision key. Dirty, mixed, missing, or stale data
   * requests exact tracing, so the accelerator cannot introduce light leaks or history.
   */
  resolve(plan: SvoStaticShadowFieldPlan, page: SvoNodeMipPageKey | string, lightId: number): SvoStaticShadowRenderDecision {
    const owner = this.candidate?.plan.cacheKey === plan.cacheKey
      ? this.candidate
      : this.visible?.plan.cacheKey === plan.cacheKey ? this.visible : undefined;
    if (!owner) return decision("unknown");
    const pagePlan = owner.pageByKey.get(pageKeyString(page));
    const channel = owner.channelByLightId.get(lightId);
    if (!pagePlan || !channel) return decision("unknown");
    const [visibleMask, occludedMask] = masks(owner, pagePlan.slot);
    return decision(certificateFromMasks(visibleMask, occludedMask, channel.channel));
  }

  candidateSnapshot(): SvoStaticShadowFieldSnapshot | undefined {
    return this.candidate && snapshot(this.candidate);
  }

  /** Deterministic page-major/light-index work for a bounded asynchronous builder. */
  dirtyWork(maximumWork = Number.MAX_SAFE_INTEGER): readonly SvoStaticShadowBuildWork[] {
    if (!Number.isSafeInteger(maximumWork) || maximumWork < 0) {
      throw new RangeError("SVO static-shadow work limit must be a non-negative safe integer");
    }
    if (!this.candidate || maximumWork === 0) return [];
    const work: SvoStaticShadowBuildWork[] = [];
    for (const page of this.candidate.plan.pages) {
      const [visibleMask, occludedMask] = masks(this.candidate, page.slot);
      const readyMask = (visibleMask | occludedMask) >>> 0;
      for (const light of this.candidate.plan.lightChannels) {
        if ((readyMask & bit(light.channel)) !== 0) continue;
        work.push({
          cacheKey: this.candidate.plan.cacheKey,
          pageKey: page.key,
          pageKeyString: page.keyString,
          slot: page.slot,
          lightId: light.lightId,
          lightRevision: light.lightRevision,
          lightChannel: light.channel,
        });
        if (work.length === maximumWork) return work;
      }
    }
    return work;
  }

  visibleSnapshot(): SvoStaticShadowFieldSnapshot | undefined {
    return this.visible && snapshot(this.visible);
  }

  /** Promotes only a fully certified exact-revision field. */
  publish(): SvoStaticShadowFieldPublicationDecision {
    if (!this.candidate) {
      if (this.visible) return { published: true, reason: "already-visible", visible: snapshot(this.visible) };
      return { published: false, reason: "no-candidate" };
    }
    if (this.visible) {
      const candidatePlan = this.candidate.plan;
      const visiblePlan = this.visible.plan;
      if (candidatePlan.sourceGeneration < visiblePlan.sourceGeneration
          || candidatePlan.lightRevision < visiblePlan.lightRevision) {
        return { published: false, reason: "revision-order", visible: snapshot(this.visible) };
      }
      if (candidatePlan.sourceGeneration === visiblePlan.sourceGeneration
          && candidatePlan.lightRevision === visiblePlan.lightRevision
          && candidatePlan.cacheKey !== visiblePlan.cacheKey) {
        return { published: false, reason: "revision-collision", visible: snapshot(this.visible) };
      }
    }
    const required = this.candidate.plan.requiredLightMask;
    for (const page of this.candidate.plan.pages) {
      const [visibleMask, occludedMask] = masks(this.candidate, page.slot);
      if (((visibleMask | occludedMask) >>> 0) !== required) {
        return { published: false, reason: "dirty-pages", visible: this.visible && snapshot(this.visible) };
      }
    }
    this.visible = this.candidate;
    this.candidate = undefined;
    return { published: true, reason: "published", visible: snapshot(this.visible) };
  }

  telemetry(): SvoStaticShadowFieldTelemetry {
    const owner = this.candidate ?? this.visible;
    if (!owner) {
      return {
        residentPages: 0, readyPages: 0, dirtyPages: 0, readyChannels: 0, dirtyChannels: 0,
        allocatedBytes: 0, fallback: "exact-current-frame-trace",
      };
    }
    let readyPages = 0;
    let readyChannels = 0;
    for (const page of owner.plan.pages) {
      const [visibleMask, occludedMask] = masks(owner, page.slot);
      const readyMask = (visibleMask | occludedMask) >>> 0;
      if (readyMask === owner.plan.requiredLightMask) readyPages += 1;
      readyChannels += popcount(readyMask);
    }
    const totalChannels = owner.plan.residentPageCount * owner.plan.lightChannels.length;
    return {
      visibleCacheKey: this.visible?.plan.cacheKey,
      candidateCacheKey: this.candidate?.plan.cacheKey,
      residentPages: owner.plan.residentPageCount,
      readyPages,
      dirtyPages: owner.plan.residentPageCount - readyPages,
      readyChannels,
      dirtyChannels: totalChannels - readyChannels,
      allocatedBytes: (this.visible?.plan.allocatedBytes ?? 0) + (this.candidate?.plan.allocatedBytes ?? 0),
      fallback: this.candidate || !this.visible ? "exact-current-frame-trace" : "none",
    };
  }
}

/** Fixed ABI helper; it compiles once for every light count and scene revision. */
export const svoStaticShadowFieldWGSL = /* wgsl */ `
struct SvoStaticShadowPage {
  visibleMask:u32,
  occludedMask:u32,
}
const SVO_STATIC_SHADOW_UNKNOWN:u32=0u;
const SVO_STATIC_SHADOW_VISIBLE:u32=1u;
const SVO_STATIC_SHADOW_OCCLUDED:u32=2u;
const SVO_STATIC_SHADOW_MIXED:u32=3u;
fn svoStaticShadowCertificate(page:SvoStaticShadowPage,lightIndex:u32)->u32{
  let lightBit=1u<<lightIndex;
  let visible=select(0u,1u,(page.visibleMask&lightBit)!=0u);
  let occluded=select(0u,2u,(page.occludedMask&lightBit)!=0u);
  return visible|occluded;
}
`;
