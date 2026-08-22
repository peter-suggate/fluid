/** Four cells on each axis are executed by one rung-major transport packet. */
export const SPARSE_CM12_HOME_FRAME_PACKET_EDGE = 4;
/** Phase 1 permits a clamped halo radius of at most three home-rung cells. */
export const SPARSE_CM12_HOME_FRAME_HALO_MAX_RADIUS = 3;
export const SPARSE_CM12_HOME_FRAME_HALO_MAX_EDGE: 10 =
  (SPARSE_CM12_HOME_FRAME_PACKET_EDGE
    + 2 * SPARSE_CM12_HOME_FRAME_HALO_MAX_RADIUS) as 10;
export const SPARSE_CM12_HOME_FRAME_HALO_CAPACITY: 1000 =
  SPARSE_CM12_HOME_FRAME_HALO_MAX_EDGE ** 3 as 1000;
export const SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_SIZE = 64;

/**
 * The cache carries one u32 cell ordinal and one native vec4 velocity per
 * lattice site. WGSL lays the arrays out separately, so both arrays retain
 * their natural alignment and consume exactly 20 bytes per site.
 */
export const SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_ARRAY_BYTES =
  SPARSE_CM12_HOME_FRAME_HALO_CAPACITY * (4 + 16);
/** Conservative allowance for the scalar/vector cache metadata. */
export const SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_METADATA_BYTES = 64;
export const SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_BYTES =
  SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_ARRAY_BYTES
  + SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_METADATA_BYTES;

export const SPARSE_CM12_HOME_FRAME_HALO_WGSL_API = Object.freeze({
  stage: "cm12StageTransportHomeFrameHalo",
  lookupAtFine: "cm12HomeHaloLookupAtFine",
  sampleVelocity: "cm12HomeHaloSampleVelocity",
  stencil: "cm12HomeHaloTransportStencil",
  traceCharacteristic: "cm12HomeHaloTraceCharacteristic",
  traceDeparture: "cm12HomeHaloTraceDeparture",
  traceArrival: "cm12HomeHaloTraceArrival",
} as const);

export interface SparseCM12TransportHomeFrameHaloLayout {
  readonly packetEdge: 4;
  readonly maximumRadius: 1 | 2 | 3;
  readonly maximumEdge: number;
  readonly capacity: number;
  readonly workgroupSize: 64;
  readonly cellOrdinalBytes: number;
  readonly effectiveVelocityBytes: number;
  readonly metadataBytes: 64;
  readonly totalWorkgroupBytes: number;
}

export function createSparseCM12TransportHomeFrameHaloLayout(
  maximumRadius: 1 | 2 | 3 = SPARSE_CM12_HOME_FRAME_HALO_MAX_RADIUS,
): SparseCM12TransportHomeFrameHaloLayout {
  const maximumEdge = SPARSE_CM12_HOME_FRAME_PACKET_EDGE + 2 * maximumRadius;
  const capacity = maximumEdge ** 3;
  const cellOrdinalBytes = 4 * capacity;
  const effectiveVelocityBytes = 16 * capacity;
  const totalWorkgroupBytes = cellOrdinalBytes + effectiveVelocityBytes
    + SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_METADATA_BYTES;
  if (totalWorkgroupBytes > 32 * 1024) {
    throw new Error("transport home-frame halo exceeds 32 KiB workgroup storage");
  }
  return Object.freeze({
    packetEdge: SPARSE_CM12_HOME_FRAME_PACKET_EDGE,
    maximumRadius,
    maximumEdge,
    capacity,
    workgroupSize: SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_SIZE,
    cellOrdinalBytes,
    effectiveVelocityBytes,
    metadataBytes: SPARSE_CM12_HOME_FRAME_HALO_WORKGROUP_METADATA_BYTES,
    totalWorkgroupBytes,
  });
}
