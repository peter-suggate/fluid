/**
 * Matrix-free composite pressure apply for the frozen adaptive-mass milestone.
 *
 * The uploaded records are a lossless f32 packing of the CPU authority's
 * gradient rows. Each invocation owns one output cell and walks only the rows
 * incident on that cell, so the shader needs neither a dense matrix nor f32
 * atomics. Both sides of a coarse/fine seam therefore consume the same row,
 * coefficient, and dual weight: symmetry is structural rather than repaired
 * after assembly.
 */
export const twoTilePressureOperatorWGSL = /* wgsl */ `
struct Parameters {
  cellCount: u32,
  gradientRowCount: u32,
  termCount: u32,
  incidenceCount: u32,
}

struct GradientRow {
  termOffset: u32,
  termCount: u32,
  dualWeight: f32,
  reserved: u32,
}

struct GradientTerm {
  cellId: u32,
  coefficient: f32,
}

struct CellIncidence {
  rowId: u32,
  termIndex: u32,
}

@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read> gradientRows: array<GradientRow>;
@group(0) @binding(2) var<storage, read> gradientTerms: array<GradientTerm>;
@group(0) @binding(3) var<storage, read> cellIncidenceOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> cellIncidences: array<CellIncidence>;
@group(0) @binding(5) var<storage, read> pressure: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn applyTwoTilePressureOperator(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let cellId = globalId.x;
  if (cellId >= parameters.cellCount) {
    return;
  }

  var result = 0.0;
  let incidenceEnd = cellIncidenceOffsets[cellId + 1u];
  var incidenceIndex = cellIncidenceOffsets[cellId];
  while (incidenceIndex < incidenceEnd) {
    let incidence = cellIncidences[incidenceIndex];
    let row = gradientRows[incidence.rowId];

    var gradient = 0.0;
    let termEnd = row.termOffset + row.termCount;
    var termIndex = row.termOffset;
    while (termIndex < termEnd) {
      let term = gradientTerms[termIndex];
      gradient += term.coefficient * pressure[term.cellId];
      termIndex += 1u;
    }

    let ownCoefficient = gradientTerms[incidence.termIndex].coefficient;
    result += ownCoefficient * row.dualWeight * gradient;
    incidenceIndex += 1u;
  }
  output[cellId] = result;
}
`;
