export const PHYSICS_WASM_PROTOCOL_VERSION = 1 as const;
export const PHYSICS_WASM_PUBLICATION_SLOTS = 3 as const;

export type PhysicsDimension = 2 | 3;
export type PhysicsWasmArtifact = "scalar" | "simd" | "threaded";

export interface PhysicsRevision {
  readonly schemaVersion: number;
  readonly dimension: PhysicsDimension;
  readonly runEpoch: number;
  readonly commandSequence: number;
  readonly frame: number;
  readonly time: number;
  readonly injections: number;
  readonly topologyGeneration: number;
  readonly fieldRevision: number;
  readonly surfaceRevision: number;
  readonly memoryEpoch: number;
}

export type PhysicsCommandReceipt = PhysicsRevision & Readonly<Record<string, unknown>>;

export type PhysicsWorkerRequest =
  | { readonly type: "initialize"; readonly requestId: number; readonly moduleUrl: string;
      readonly artifact: PhysicsWasmArtifact; readonly threadCount: number }
  | { readonly type: "load"; readonly requestId: number; readonly sequence: number;
      readonly runEpoch: number; readonly sceneJson: string; readonly optionsJson: string }
  | { readonly type: "advance"; readonly requestId: number; readonly sequence: number;
      readonly runEpoch: number; readonly dt_s: number; readonly viewMask: number }
  | { readonly type: "apply-command"; readonly requestId: number; readonly sequence: number;
      readonly runEpoch: number; readonly commandJson: string; readonly viewMask?: number }
  | { readonly type: "snapshot"; readonly requestId: number; readonly sequence: number;
      readonly runEpoch: number; readonly viewMask: number }
  | { readonly type: "release-publication"; readonly publicationId: number;
      readonly buffer: ArrayBuffer }
  | { readonly type: "dispose"; readonly requestId: number };

export type PhysicsWorkerResponse =
  | { readonly type: "initialized"; readonly requestId: number;
      readonly artifact: PhysicsWasmArtifact; readonly threadCount: number }
  | { readonly type: "receipt"; readonly requestId: number; readonly receipt: PhysicsCommandReceipt }
  | { readonly type: "publication"; readonly requestId: number; readonly publicationId: number;
      readonly revision: PhysicsRevision; readonly byteLength: number; readonly buffer: ArrayBuffer }
  | { readonly type: "disposed"; readonly requestId: number }
  | { readonly type: "request-failed"; readonly requestId: number; readonly message: string };

export interface PhysicsPublication {
  readonly id: number;
  readonly revision: PhysicsRevision;
  readonly bytes: Uint8Array;
  release(): void;
}

export function parsePhysicsReceipt(json: string): PhysicsCommandReceipt {
  const value: unknown = JSON.parse(json);
  if (!value || typeof value !== "object") throw new TypeError("Physics Wasm returned a non-object receipt");
  const receipt = value as Record<string, unknown>;
  for (const key of ["schemaVersion", "dimension", "runEpoch", "commandSequence", "frame",
    "time", "injections", "topologyGeneration", "fieldRevision", "surfaceRevision", "memoryEpoch"] as const) {
    if (typeof receipt[key] !== "number" || !Number.isFinite(receipt[key])) {
      throw new TypeError(`Physics Wasm receipt has invalid ${key}`);
    }
  }
  if (receipt.dimension !== 2 && receipt.dimension !== 3) {
    throw new TypeError("Physics Wasm receipt has invalid dimension");
  }
  return receipt as PhysicsCommandReceipt;
}
