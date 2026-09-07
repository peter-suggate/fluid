/** Contracts identify representation and lifetime; hosts own concrete GPU resources. */
export interface PublicationPort<Value = unknown> {
  readonly id: string;
  readonly representation: string;
  readonly lifetime: "device" | "scene" | "generation" | "frame";
  readonly unit?: string;
  /** Type-only payload witness. Never invoked or included in serialized configuration. */
  readonly payload?: (value: Value) => Value;
}
export interface PortRequirement {
  readonly port: PublicationPort<any>;
  /** Explicit provider feature ID. No import-order or first-provider fallback. */
  readonly provider: string;
}
export function publicationPort<Value>(
  definition: Omit<PublicationPort<Value>, "payload">,
): PublicationPort<Value> {
  return Object.freeze({ ...definition });
}
export interface AcceptedPublication<Value> {
  readonly port: PublicationPort<Value>;
  readonly owner: string;
  readonly generation: number;
  readonly value: Value;
}

/** A consumer can never reinterpret a publication of a different representation. */
export function readPublication<Value>(
  required: PublicationPort<Value>,
  publication: AcceptedPublication<Value>,
  expected: { readonly owner: string; readonly generation: number },
): Value {
  if (!portsMatch(required, publication.port)) throw new Error(`Publication contract mismatch for ${required.id}`);
  if (publication.owner !== expected.owner) throw new Error(`Wrong publication owner for ${required.id}: expected ${expected.owner}, got ${publication.owner}`);
  if (publication.generation !== expected.generation) throw new Error(`Stale publication ${required.id}: expected ${expected.generation}, got ${publication.generation}`);
  return publication.value;
}
export function portsMatch(a: PublicationPort<any>, b: PublicationPort<any>): boolean {
  return a.id === b.id && a.representation === b.representation && a.lifetime === b.lifetime && a.unit === b.unit;
}
