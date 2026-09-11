import type { SparseCM12EffectiveTransportVelocityLayout } from
  "./sparse-cm12-effective-transport-velocity";

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

export interface SparseCM12EffectiveTransportVelocityWGSLOptions {
  readonly layout: SparseCM12EffectiveTransportVelocityLayout;
  /** Existing `array<vec4f>` storage binding; Phase 1 maps binding 3 here. */
  readonly planeName?: string;
}

/**
 * Binding-free helpers for the persistent effective velocity plane.
 *
 * The resident composes this only for the Phase-1 arm. The corresponding bind
 * groups map binding 3 (normally diagnostics `partials`) to a dedicated GPU
 * buffer for VEX commit, collocation, and conservative transport. This yields
 * one native vec4 load per sample without adding an eleventh storage binding.
 */
export function createSparseCM12EffectiveTransportVelocityWGSL(
  options: SparseCM12EffectiveTransportVelocityWGSLOptions,
): string {
  const plane = identifier(options.planeName ?? "partials", "planeName");
  return /* wgsl */ `
const CM12_EFFECTIVE_TRANSPORT_VELOCITY_CAPACITY:u32=${options.layout.cellCapacity}u;

fn cm12EffectiveTransportVelocity(cell:u32)->vec4f{
  return ${plane}[cell];
}
fn cm12PublishVexAcceptedEffectiveVelocity(cell:u32,value:vec4f){
  if(cell<CM12_EFFECTIVE_TRANSPORT_VELOCITY_CAPACITY){${plane}[cell]=value;}
}
fn cm12PublishCollocatedWetEffectiveVelocity(cell:u32,velocity:vec3f,wet:bool){
  if(wet&&cell<CM12_EFFECTIVE_TRANSPORT_VELOCITY_CAPACITY){
    ${plane}[cell]=vec4f(velocity,1.0);
  }
}
fn cm12PublishTransferredEffectiveVelocity(cell:u32,velocity:vec3f){
  if(cell<CM12_EFFECTIVE_TRANSPORT_VELOCITY_CAPACITY){
    ${plane}[cell]=vec4f(velocity,1.0);
  }
}

`;
}
