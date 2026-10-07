/** Shared MAC acceptance rule. A zero relative fraction preserves the strict
 * absolute-only reference path. APIC uses Uniform Geometric's 0.1 fraction
 * and 1e-4 s^-1 floor with a 5 s^-1 absolute ceiling. */
export const MAC_PRESSURE_RELATIVE_FLOOR = 1e-4;
export const macPressureTargetWGSL = /* wgsl */ `
fn pressureTarget(initial:f32,absolute:f32,relative:f32)->f32{
 if(relative<=0.0){return absolute;}
 return min(absolute,max(relative*initial,${MAC_PRESSURE_RELATIVE_FLOOR}));
}
`;
