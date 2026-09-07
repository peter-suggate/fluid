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
export function portsMatch(a: PublicationPort<any>, b: PublicationPort<any>): boolean {
  return a.id === b.id && a.representation === b.representation && a.lifetime === b.lifetime && a.unit === b.unit;
}
