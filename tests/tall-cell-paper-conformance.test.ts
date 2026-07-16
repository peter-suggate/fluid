import assert from "node:assert/strict";
import test from "node:test";
import { tallCellComputeShader } from "../lib/tall-cell-kernels";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { tallCellExtrapolationShader } from "../lib/tall-cell-extrapolation";
import { tallCellMultigridShaderForTests } from "../lib/tall-cell-multigrid";

test("tall level set uses Eq 4 point samples and Eq 5 reconstruction", () => {
  assert.match(tallCellComputeShader, /fn pointSamplePhi/);
  assert.match(tallCellComputeShader, /fn phiCell/);
  assert.match(tallCellComputeShader, /mix\(textureLoad\(volumeIn,vec3i\(q\.x,0,q\.z\),0\)\.x,textureLoad\(volumeIn,vec3i\(q\.x,1,q\.z\),0\)\.x,t\)/);
  assert.match(tallCellComputeShader, /fn samplePhi[\s\S]*phiCell\(b\+vec3i\(1,1,1\)\)/);
  assert.match(tallCellComputeShader, /pointSamplePhi\(id\)<=0\.0/);
});

test("phi is semi-Lagrangian while velocity retains bounded MacCormack", () => {
  assert.match(tallCellComputeShader, /var phi=samplePhi\(traceDeparture\(p,dt\)\)/);
  assert.match(tallCellComputeShader, /var v=boundedMacCormack\(id,p\)/);
  assert.match(tallCellComputeShader, /corrected=predicted\+0\.5\*\(original-reversed\)/);
  assert.doesNotMatch(tallCellComputeShader, /correctedPhi|phiCorrector|boundedMacCormackPhi/);
});

test("level-set reinitialization keeps the paper safeguards", () => {
  assert.match(tallCellComputeShader, /fn clampPhi/);
  assert.match(tallCellComputeShader, /fn reinitializePhi/);
  assert.match(tallCellComputeShader, /adjacentToInterface\(q,current\)/);
  assert.match(tallCellComputeShader, /abs\(current\)>3\.0\*cell/);
  assert.match(tallCellComputeShader, /5\.0\*cell/);
  assert.match(tallCellComputeShader, /clamp\(candidate,current-cell,current\+cell\)/);
});

test("pressure and projection use phi ghost fractions and endpoint divergence", () => {
  assert.match(tallCellComputeShader, /fn interfaceFraction\(a:f32,b:f32\)->f32\{return clamp\(abs\(a\)\/max\(abs\(a\)\+abs\(b\)/);
  assert.match(tallCellComputeShader, /let wet=activeSample\(id\)&&pointSamplePhi\(id\)<=0\.0/);
  assert.match(tallCellComputeShader, /fn divergenceAt\(id:vec3i\)->f32\{\s*return pointDivergenceAt\(vec3i\(floor\(samplePoint\(id\)\)\)\);/);
  assert.match(tallCellComputeShader, /v\[axis\]-=scale\*pressureGradientAt\(q,axis\)/);
  assert.doesNotMatch(tallCellComputeShader, /storeControlVolumeDivergence/);
  assert.doesNotMatch(tallCellComputeShader, /1\.0\/\(distance\*distance\)/);
});

test("level-set remesh follows zero crossings and uses least-squares endpoint transfer", () => {
  assert.match(tallCellComputeShader, /if\(\(previous<=0\.0\)!=\(current<=0\.0\)\)/);
  assert.match(tallCellComputeShader, /fn leastSquaresPhi/);
  assert.match(tallCellComputeShader, /fn leastSquaresVelocity/);
  assert.match(tallCellComputeShader, /phi=select\(select\(fit\.x,fit\.y,id\.y==1\),fit\.y,crossing\)/);
  assert.match(tallCellComputeShader, /let crossing=\(fit\.x<=0\.0\)!=\(fit\.y<=0\.0\)/);
  assert.doesNotMatch(tallCellComputeShader, /desired=max\(desired,i32\(ceil\(columnWaterCells/);
  assert.doesNotMatch(tallCellComputeShader, /wetTopFloor/);
  assert.match(tallCellComputeShader, /base=min\(base,nextColumnBases\[u32\(q\.x\+d\.x\*q\.y\)\]\+delta\)/);
  assert.match(tallCellComputeShader, /bodyUpper=min\(bodyUpper,predictedBottom\)/);
});

test("global volume correction is bounded to interface samples", () => {
  assert.match(tallCellComputeShader, /fn volumeCorrectionDivergence/);
  assert.match(tallCellComputeShader, /abs\(pointSamplePhi\(id\)\)<band/);
  assert.match(tallCellComputeShader, /divergenceAt\(id\)-volumeCorrectionDivergence\(id\)/);
});

test("restricted level-set path has retired its VOF transport machinery", () => {
  assert.doesNotMatch(tallCellComputeShader, /rawVolumeFlux|advectedTallVolume|sharpenCompute|sharpenScatter|sharpenResolve/);
  assert.doesNotMatch(tallCellComputeShader, /sharpenDeposits|transportVolumeIn|tallStoreAlpha|columnWaterCells/);
});

test("uniform VOF keeps mass-conserving density sharpening", () => {
  assert.match(legacyUniformComputeShader, /fn sharpenDeltaRho/);
  assert.match(legacyUniformComputeShader, /fn sharpenCompute/);
  assert.match(legacyUniformComputeShader, /fn sharpenScatter/);
  assert.match(legacyUniformComputeShader, /fn sharpenResolve/);
  assert.match(legacyUniformComputeShader, /let deltaT=3\.0\*params\.dimsDt\.w;let tau=0\.4/);
  assert.match(legacyUniformComputeShader, /-deltaRho\*1048576\.0/);
});

test("multigrid coarse levels never allocate degenerate base-1 stores", () => {
  assert.match(tallCellMultigridShaderForTests, /if\(upper>=2\)\{b=max\(b,2\);\}/);
});

test("hierarchical extrapolation follows paper Sec 3.3.1", () => {
  assert.match(tallCellExtrapolationShader, /fn downsampleExtrapolationBase/);
  assert.match(tallCellExtrapolationShader, /clamp\(\(maximum\+1\)\/2,0,max\(0,destinationFineY\(\)-layers\)\)/);
  assert.match(tallCellExtrapolationShader, /fn downsampleVelocity/);
  assert.match(tallCellExtrapolationShader, /if\(sample\.w>0\.5\)\{sum\+=sample\.xyz;known\+=1\.0;\}/);
  assert.match(tallCellExtrapolationShader, /if\(current\.w>0\.5\)\{textureStore\(destinationVelocity,id,current\);return;\}/);
  assert.match(tallCellExtrapolationShader, /if\(weight>1e-8\)/);
});
