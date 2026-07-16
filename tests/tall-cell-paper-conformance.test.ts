import assert from "node:assert/strict";
import test from "node:test";
import { tallCellComputeShader } from "../lib/tall-cell-kernels";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { tallCellExtrapolationShader } from "../lib/tall-cell-extrapolation";
import { tallCellMultigridShaderForTests } from "../lib/tall-cell-multigrid";

// Structural conformance checks for the stages added in the 2026-07-15 paper
// audit (docs/TALL_CELLS_PAPER.md Appendices A-C). They pin the presence and
// shape of each fix so a refactor cannot silently drop one; behavioral gates
// live in the GPU smoke suite (test:webgpu:dam-tall-active).

test("tall occupancy is the constant store density, never a settled sub-cell interface", () => {
  assert.match(tallCellComputeShader, /fn tallFillCells/);
  assert.match(tallCellComputeShader, /fn tallStoreAlpha/);
  assert.match(tallCellComputeShader, /fn pointSampleAlpha/);
  // The 2026-07-16 audit: reconstructing a settled fill (clamp(fill - y))
  // re-teleported band water to the column floor every step and pumped
  // kinetic energy through the phantom-gap collapse. Every consumer now
  // shares the store density.
  assert.doesNotMatch(tallCellComputeShader, /clamp\(fill-f32\(q\.y\)/);
  assert.match(tallCellComputeShader, /if\(id\.y<=1\)\{return tallStoreAlpha\(id\.x,id\.z\);\}/);
  // Gravity uses the MAC face gate (either adjacent cell liquid, matching
  // the uniform path); pressure RHS, projection wetness, extrapolation
  // seeding, and the diagnostics gate on the point sample.
  assert.match(tallCellComputeShader, /if\(pointAlpha>=0\.5\|\|volumeCell\(q\+vec3i\(0,1,0\)\)>=0\.5\)\{v\.y\+=params\.cellGravity\.w\*dt;\}/);
  assert.match(tallCellComputeShader, /let wet=activeSample\(id\)&&pointSampleAlpha\(id\)>=0\.5/);
  assert.match(tallCellComputeShader, /let ownAlpha=pointSampleAlpha\(id\);if\(ownAlpha<0\.5\)/);
});

test("tall cells use Eq 5 linear reconstruction and endpoint point divergence", () => {
  assert.match(tallCellComputeShader, /positiveFaceVelocity\(q,0u\)-positiveFaceVelocity\(q-vec3i\(1,0,0\),0u\)/);
  assert.match(tallCellComputeShader, /v\[axis\]-=scale\*pressureGradientAt\(q,axis\)/);
  // Eq 5: velocity inside a tall cell interpolates linearly between the
  // endpoint dofs. The piecewise-constant reconstruction plus the averaged
  // control-volume divergence admitted a whole-store free-fall mode balanced
  // by fake lateral spreading — the 2026-07-16 dam-break dome.
  assert.match(tallCellComputeShader, /fn validVelocityCell[\s\S]{0,900}mix\(textureLoad\(velocityIn,vec3i\(q\.x,0,q\.z\),0\)\.xyz,textureLoad\(velocityIn,vec3i\(q\.x,1,q\.z\),0\)\.xyz,t\)/);
  assert.doesNotMatch(tallCellComputeShader, /storeControlVolumeDivergence/);
  // Eq 13/19: divergence is a point sample at the endpoint sub-cells; the
  // bottom sample sees the closed floor face directly.
  assert.match(tallCellComputeShader, /fn divergenceAt\(id:vec3i\)->f32\{\s*return pointDivergenceAt\(vec3i\(floor\(samplePoint\(id\)\)\)\);/);
  // Bottom-endpoint row (Eq 15/16 with solid below): the top endpoint
  // couples at s/h^2 = 1/(distance*h) — the exact staggered adjoint of the
  // bottom point divergence (the old 1/distance^2 admitted the free-fall
  // mode). Top-endpoint row keeps the paper's strong coefficients (band
  // 1/h^2, bottom 1/(distance*h)); the exact adjoint anchored the top-dof
  // layer too weakly and diverged on deep scenes.
  assert.match(tallCellComputeShader, /if\(id\.y==0&&base>0\)\{[\s\S]{0,3000}1\.0\/\(distance\*h\.y\)/);
  assert.match(tallCellComputeShader, /else if\(id\.y==1\)\{[^}]{0,700}1\.0\/\(distance\*h\.y\)[^}]{0,700}1\.0\/\(h\.y\*h\.y\)/);
  assert.doesNotMatch(tallCellComputeShader, /1\.0\/\(distance\*distance\)/);
});

test("remeshing scans the full column and never strands a column's own water", () => {
  // Paper Section 8: a settled surface inside the tall region is a crossing.
  assert.match(tallCellComputeShader, /fill>=0\.5&&fill<f32\(oldBase\)-0\.5/);
  // No per-step clamp on base movement; representability outranks D.
  assert.doesNotMatch(tallCellComputeShader, /max\(0,oldBase-delta\)/);
  assert.match(tallCellComputeShader, /fn columnWaterCells/);
  assert.match(tallCellComputeShader, /desired=max\(desired,i32\(ceil\(columnWaterCells\(x,z\)\)\)-layers\)/);
  // The smoothing pass applies the same floor after the neighbor-D min.
  assert.match(tallCellComputeShader, /base=max\(base,u32\(floorBase\)\)/);
});

test("rigid proximity can only shorten a tall store", () => {
  // A rigid body is not a liquid interface. The old bodyLower/top lift made
  // the store grow as soon as a falling body entered the water's air halo.
  assert.doesNotMatch(tallCellComputeShader, /bodyLower/);
  assert.doesNotMatch(tallCellComputeShader, /top\+1-layers/);
  assert.match(tallCellComputeShader, /bodyUpper=min\(bodyUpper,predictedBottom\)/);
  assert.match(tallCellComputeShader, /desired=min\(desired,bodyUpper\)/);
});

test("remap conserves the column integral through band rescaling and overflow settling", () => {
  assert.match(tallCellComputeShader, /var bandScale=1\.0;if\(oldAmount<regularAmount&&regularAmount>1e-6\)/);
  assert.match(tallCellComputeShader, /let overflow=max\(0\.0,residual-f32\(newBase\)\)/);
  assert.match(tallCellComputeShader, /alpha=\(residual-min\(overflow,capacity\)\)\/f32\(max\(newBase,1\)\)/);
});

test("overfull stores drain through the mass-conserving paper's correction divergence", () => {
  assert.match(tallCellComputeShader, /fn volumeCorrectionDivergence/);
  // lambda = 0.5, eta = 1, expressed against the paper's 1/30 s step.
  assert.match(tallCellComputeShader, /min\(0\.5\*excess,1\.0\)\*30\.0/);
  // Sec 3.7 works in DENSITY units for the tall store too: scaling the
  // excess or deficit by base made the correction up to base times the
  // paper rate, an artificial source strong enough to pump energy.
  assert.doesNotMatch(tallCellComputeShader, /alpha-1\.0\)\*f32\(baseAt/);
  // The mirrored refill branch keeps a submerged partial store from
  // drifting through the 0.5 classification cliff (and it damps, rather
  // than feeds, the late-slosh multigrid event; see the 2026-07-16 audit).
  assert.match(tallCellComputeShader, /-min\(0\.5\*deficit,1\.0\)\*30\.0/);
  // Subtracted on the RHS so projection leaves div_new = +c (outward drain).
  assert.match(tallCellComputeShader, /divergenceAt\(id\)-volumeCorrectionDivergence\(id\)/);
  assert.doesNotMatch(tallCellComputeShader, /divergenceAt\(id\)\+volumeCorrectionDivergence\(id\)/);
});

for (const [label, shader, tau] of [["tall-cell", tallCellComputeShader, "0.45"], ["uniform", legacyUniformComputeShader, "0.4"]] as const) {
  test(`${label} shader implements the Sec 3.5 density sharpening stage`, () => {
    assert.match(shader, /fn sharpenDeltaRho/);
    assert.match(shader, /fn sharpenCompute/);
    assert.match(shader, /fn sharpenScatter/);
    assert.match(shader, /fn sharpenResolve/);
    // Paper parameters: fictitious step 3*dt, epsilon = 1e-5, and trace
    // distance D = 2.1 cells. The paper permits larger tau for cohesion; the
    // tall path uses 0.45 to offset remap diffusion, uniform keeps 0.4.
    assert.match(shader, new RegExp(`let deltaT=3\\.0\\*params\\.dimsDt\\.w;let tau=${tau.replace(".", "\\.")}`));
    assert.match(shader, /rho\+deltaRho<0\.0\|\|rho<1e-5/);
    assert.match(shader, /let maximumDistance=2\.1/);
    // Eq 14 weight and the Eq 17 liquid-side guard (mass only moves from the
    // air side to the liquid side).
    assert.match(shader, /\(rho-0\.5\)\*\(rho-0\.5\)\*\(rho-0\.5\)\*\(1\.0-min\(1\.0,maximumDifference\/tau\)\)/);
    assert.match(shader, /else if\(rho>0\.5\)\{deltaRho=0\.0;\}/);
    // The scatter deposits exactly the removed mass in fixed point at 2^-20
    // resolution so dust below the Eq 17 epsilon still round-trips.
    assert.match(shader, /atomicAdd\(&sharpenDeposits\[u32\(indices\[corner\]\)\],i32\(round\(/);
    assert.match(shader, /-deltaRho\*1048576\.0/);
  });
}

test("tall sharpening deposits into the packed layout with tall-average scaling", () => {
  assert.match(tallCellComputeShader, /fn packedDepositIndex/);
  // Deposits carry mass; the resolve pass converts a tall bottom store's
  // mass to its average AFTER quantization so contributions cannot round
  // away (average * base = mass).
  assert.match(tallCellComputeShader, /if\(id\.y==0\)\{deposit\/=f32\(max\(baseAt\(id\.x,id\.z\),1\)\);\}/);
  // Band corners without capacity are skipped so the advection clamp cannot
  // destroy deposited mass.
  assert.match(tallCellComputeShader, /if\(!insideTall&&w>0\.0&&volumeCell\(destination\)>=1\.0\)\{w=0\.0;\}/);
  assert.match(legacyUniformComputeShader, /if\(w>0\.0&&volume\(destination\)>=1\.0\)\{w=0\.0;\}/);
  assert.match(legacyUniformComputeShader, /fn volumeCorrectionDivergence/);
});

test("multigrid coarse levels never allocate degenerate base-1 stores", () => {
  // Eq 9 halving turns fine base 2 into coarse base 1, where BOTH endpoint
  // dofs land on the same world cell (sampleY 0.5). For shallow tanks that
  // duplication covers whole coarse levels and the coarse correction
  // re-injects a floor-row checkerboard the smoother cannot remove — the
  // still-tank blow-up at ~1.4 s (2026-07-16 audit). Coarse columns must own
  // a genuine h >= 2 tall cell whenever the level can represent one.
  assert.match(tallCellMultigridShaderForTests, /if\(upper>=2\)\{b=max\(b,2\);\}/);
});

test("hierarchical extrapolation follows paper Sec 3.3.1", () => {
  // Eq 9 base downsampling (ceil of half the 2x2 maximum) and the two sweeps.
  assert.match(tallCellExtrapolationShader, /fn downsampleExtrapolationBase/);
  assert.match(tallCellExtrapolationShader, /clamp\(\(maximum\+1\)\/2,0,max\(0,destinationFineY\(\)-layers\)\)/);
  assert.match(tallCellExtrapolationShader, /fn downsampleVelocity/);
  // Fine-to-coarse: known when any covered fine sample is known.
  assert.match(tallCellExtrapolationShader, /if\(sample\.w>0\.5\)\{sum\+=sample\.xyz;known\+=1\.0;\}/);
  // Coarse-to-fine: unknown samples fill by renormalized interpolation and
  // known samples pass through unchanged.
  assert.match(tallCellExtrapolationShader, /if\(current\.w>0\.5\)\{textureStore\(destinationVelocity,id,current\);return;\}/);
  assert.match(tallCellExtrapolationShader, /if\(weight>1e-8\)/);
});
