/**
 * Operator-only GPU execution for the frozen two-tile conservative transport.
 *
 * Trace construction, CM12 conditioning, volume-scaled deficit return, and
 * nextGamma construction remain in the CPU oracle for this milestone. The GPU
 * consumes those receiver CSR rows verbatim. Persistent gamma deliberately is
 * not multiplied by the density matrix: it is the authoritative nextGamma
 * state produced while the rows were built.
 */
export const twoTileConservativeTransportWGSL = /* wgsl */ `
struct Parameters {
  receiverCount: u32,
  coefficientCount: u32,
  reserved0: u32,
  reserved1: u32,
}

struct TransportCoefficient {
  donorCellId: u32,
  coefficient: f32,
}

@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read> rowOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> coefficients: array<TransportCoefficient>;
@group(0) @binding(3) var<storage, read> authoritativeNextGamma: array<f32>;
@group(0) @binding(4) var<storage, read> sourceDensity: array<f32>;
@group(0) @binding(5) var<storage, read_write> destinationDensity: array<f32>;
@group(0) @binding(6) var<storage, read_write> destinationGamma: array<f32>;

@compute @workgroup_size(64)
fn applyTwoTileConservativeTransport(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let receiverCellId = globalId.x;
  if (receiverCellId >= parameters.receiverCount) {
    return;
  }

  var density = 0.0;
  let coefficientEnd = rowOffsets[receiverCellId + 1u];
  var coefficientIndex = rowOffsets[receiverCellId];
  while (coefficientIndex < coefficientEnd) {
    let entry = coefficients[coefficientIndex];
    density += entry.coefficient * sourceDensity[entry.donorCellId];
    coefficientIndex += 1u;
  }
  destinationDensity[receiverCellId] = density;
  destinationGamma[receiverCellId] = authoritativeNextGamma[receiverCellId];
}
`;
