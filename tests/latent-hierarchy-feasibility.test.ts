import assert from "node:assert/strict";
import test from "node:test";

// Independent mathematical counterexamples. These do not import a candidate
// implementation and do not authorize changing a density or its target mass.
const beta = (x: number) => 30 * x * x * (1 - x) ** 2;
const betaDerivative = (x: number) => 60 * x * (1 - x) * (1 - 2 * x);
const smoothCap = (t: number) => (1 - t) ** 3 * (1 + 3 * t);
const capDerivative = (t: number) => -12 * t * (1 - t) ** 2;
const capSecondDerivative = (t: number) => -12 + 48 * t - 36 * t * t;
const currentNormal = (x: number) => smoothCap(x <= .5 ? 2 * x : 2 - 2 * x);
const reconstructedNormal = (x: number) => 1 - 3 / 5 * beta(x);
const reconstructedTopTrace = (x: number) => 1 - 8 / 15 * beta(x);
const close = (actual: number, expected: number, tolerance = 2e-14) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);

// Offset from the dry threshold w/2. Bottom trace=0, top trace=T(x),
// both y-derivative traces=N(x). This is the actual tensor Hermite face
// interpolant before adding its single scalar beta(x)*beta(y) coefficient.
function candidateOffset(x: number, y: number, delta: number): number {
  const normal = reconstructedNormal(x), top = reconstructedTopTrace(x);
  return y * normal + (3 * y * y - 2 * y ** 3) * (top - normal) + delta * beta(x) * beta(y);
}
function candidateNormal(x: number, y: number, delta: number): number {
  const normal = reconstructedNormal(x), top = reconstructedTopTrace(x);
  return normal + (6 * y - 6 * y * y) * (top - normal) + delta * beta(x) * betaDerivative(y);
}

test("the transported face is identically dry with a C2 piecewise-quartic normal trace", () => {
  // S(t)=1-6t²+8t³-3t⁴ >=0 on [0,1]. Its two mirrored halves
  // join with equal value, first derivative, and second derivative.
  close(smoothCap(0), 1); close(capDerivative(0), 0);
  close(smoothCap(1), 0); close(2 * capDerivative(1), -2 * capDerivative(1));
  close(4 * capSecondDerivative(1), 0);
  const capIntegral = 1 - 6 / 3 + 8 / 4 - 3 / 5;
  close(capIntegral, 2 / 5);
  close(.5 * capIntegral + .5 * capIntegral, 2 / 5);
  for (let i = 0; i <= 100; i++) for (let j = 0; j <= 10; j++) {
    const x = i / 100, y = j / 10;
    assert.ok(currentNormal(x) >= 0);
    // psi=w/2+y*f(x); its density is zero throughout this face.
    assert.equal(Math.max(0, Math.min(1, -y * currentNormal(x))), 0);
  }
});

test("the prescribed normal-derivative moments uniquely create a negative inward derivative", () => {
  // Endpoint values1, endpoint derivatives0 give the constant cubic1.
  // Integral(beta)=1, so the line-integral constraint2/5 fixes delta=-3/5.
  close(30 * (1 / 3 - 2 / 4 + 1 / 5), 1);
  close(reconstructedNormal(0), 1); close(reconstructedNormal(1), 1);
  close(betaDerivative(0), 0); close(betaDerivative(1), 0);
  close(1 - 3 / 5, 2 / 5);
  close(reconstructedNormal(.5), -1 / 8);
  // A dry upper trace using its transported midpoint has T(.5)=0;
  // it remains nonnegative, so the obstruction is the shared derivative.
  close(reconstructedTopTrace(.5), 0);
  for (let i = 0; i <= 100; i++) assert.ok(reconstructedTopTrace(i / 100) >= -1e-15);
});

test("no finite face bubble can change the wetting linear term or achieve zero face mass", () => {
  // For every finite delta, psi-w/2 = -y/8 + O(y²) at x=1/2.
  // The exact derivative is independent of delta, not a sampled fit.
  for (const delta of [-1024, -1, 0, 1, 1024, 1_000_000]) {
    close(candidateOffset(.5, 0, delta), 0);
    close(candidateNormal(.5, 0, delta), -1 / 8);
    close(candidateOffset(.5, 1, delta), 0);
    close(candidateNormal(.5, 1, delta), -1 / 8);

    // Stronger: a positive-area lower bound, rather than one wet ray.
    // On x in [2/5,3/5], N(x)<=-a, a=23/625. For y in [0,1],
    // the remaining terms are <=C*y², C=3/8+(225/4)*abs(delta).
    // Thus 0<y<epsilon implies q>=a*y/(2w), and the face integral
    // is >=a*epsilon²/(20w)>0. Every finite real delta has such an
    // epsilon; the finite examples below verify the stated coefficients.
    const a = 23 / 625, width = 1, C = 3 / 8 + 225 / 4 * Math.abs(delta);
    const epsilon = Math.min(.5, a / (2 * C), width / a);
    const positiveMassLowerBound = a * epsilon ** 2 / (20 * width);
    assert.ok(Number.isFinite(positiveMassLowerBound) && positiveMassLowerBound > 0);
    for (const x of [.4, .45, .5, .55, .6]) {
      assert.ok(reconstructedNormal(x) <= -a + 2e-15);
      const y = epsilon / 2, offset = candidateOffset(x, y, delta);
      assert.ok(offset <= -a * y / 2, `delta=${delta}, x=${x} must have a wet strip`);
      assert.ok(Math.max(0, Math.min(1, -offset / width)) >= a * y / (2 * width));
    }
  }
});

test("mixing midpoint and mean preferences invents a transverse latent mode in a dry extruded field", () => {
  // psi(x,y)=2+f(x), w=1 is strictly dry and independent of y.
  // A midpoint-preferred edge has mean 3-8/15, but the current mean
  // is 2+2/5. A face-mean preference therefore inserts delta=-1/15,
  // although its density constraint was already satisfied identically.
  const currentMean = 2 + 2 / 5, edgeMean = 3 - 8 / 15;
  const faceDelta = currentMean - edgeMean;
  close(faceDelta, -1 / 15);
  const edge = (x: number) => 3 - 8 / 15 * beta(x);
  const face = (x: number, y: number) => edge(x) + faceDelta * beta(x) * beta(y);
  close(face(.5, 0), 2); close(face(.5, .5), 2 - 15 / 64);
  assert.notEqual(face(.5, 0), face(.5, .5));
  for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) {
    assert.equal(Math.max(0, Math.min(1, .5 - face(i / 20, j / 20))), 0);
  }
});

test("a C1 donor seam does not supply a unique rotated mixed endpoint derivative", () => {
  // Source psi(X,Y)=X² for X>=0, 2X² otherwise is C1. A rotation
  // X=(x+y)/sqrt(2) gives d_x psi(0,y)=y or2y. The proposed shared
  // mixed jet d_y d_x psi at the seam is therefore not defined.
  const mappedGradientX = (x: number, y: number) => (x + y) * (x + y >= 0 ? 1 : 2);
  close(mappedGradientX(0, 0), 0);
  for (const epsilon of [1e-2, 1e-4, 1e-6]) {
    close(mappedGradientX(0, epsilon) / epsilon, 1);
    close(mappedGradientX(0, -epsilon) / -epsilon, 2);
  }
});
