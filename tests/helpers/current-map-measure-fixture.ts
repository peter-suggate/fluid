export interface CurrentMapMeasureFixture {
  readonly cellLowerFine: readonly number[];
  readonly controls: readonly number[];
  readonly transportedVelocityFine: readonly number[];
  readonly poolHeightFine: number;
  readonly sphereCenterFine: readonly number[];
  readonly sphereRadiusFine: number;
  readonly transitionWidthFine: number;
  readonly referenceMean: readonly number[];
}

/** CPU quadrature replay of a real candidate density-map readback. Convert
 * its cubic tensor to powers independently of production GPU evaluation.
 * Constant transported velocity isolates integration conditioning from the
 * historical trajectory method; it is an explicit QA integrand, not a
 * production velocity update. No old RK2 trace is replayed.
 */
export function currentMapMeasureFixtureSampler(fixture: CurrentMapMeasureFixture) {
  const basis = [[1,-3,3,-1],[4,0,-6,3],[1,3,3,-3],[0,0,0,1]].map(row => row.map(v => v/6));
  const coefficients = new Float64Array(192);
  for (let z = 0; z < 4; ++z) for (let y = 0; y < 4; ++y) for (let x = 0; x < 4; ++x) {
    const from = 3*(x+4*(y+4*z));
    for (let kz = 0; kz < 4; ++kz) for (let ky = 0; ky < 4; ++ky) for (let kx = 0; kx < 4; ++kx) {
      const weight = basis[x]![kx]!*basis[y]![ky]!*basis[z]![kz]!,to = 3*(kx+4*(ky+4*kz));
      for (let c = 0; c < 3; ++c) coefficients[to+c] += weight*fixture.controls[from+c]!;
    }
  }
  return (point: readonly [number,number,number]): readonly [number,number,number] => {
    const t = point.map((v,a) => v-fixture.cellLowerFine[a]!);
    const powers = t.map(v => [1,v,v*v,v*v*v]),derivatives = t.map(v => [0,1,2*v,3*v*v]);
    const displacement = [0,0,0],jacobian = [1,0,0,0,1,0,0,0,1];
    for (let z = 0; z < 4; ++z) for (let y = 0; y < 4; ++y) for (let x = 0; x < 4; ++x) {
      const weight = powers[0]![x]!*powers[1]![y]!*powers[2]![z]!;
      const dx = derivatives[0]![x]!*powers[1]![y]!*powers[2]![z]!;
      const dy = powers[0]![x]!*derivatives[1]![y]!*powers[2]![z]!;
      const dz = powers[0]![x]!*powers[1]![y]!*derivatives[2]![z]!,at = 3*(x+4*(y+4*z));
      for (let c = 0; c < 3; ++c) {
        const value = coefficients[at+c]!;displacement[c] += weight*value;
        jacobian[c] += dx*value;jacobian[3+c] += dy*value;jacobian[6+c] += dz*value;
      }
    }
    const mapped = point.map((v,a) => v+displacement[a]!);
    const j = jacobian,determinant = j[0]!*(j[4]!*j[8]!-j[5]!*j[7]!)-j[3]!*(j[1]!*j[8]!-j[2]!*j[7]!)
      +j[6]!*(j[1]!*j[5]!-j[2]!*j[4]!);
    const radius = fixture.sphereRadiusFine;
    const sphere = (mapped.reduce((sum,v,a) => sum+(v-fixture.sphereCenterFine[a]!)**2,0)-radius**2)/(2*radius);
    const phi = Math.min(mapped[1]!-fixture.poolHeightFine,sphere),raw = .5-phi/fixture.transitionWidthFine;
    const density = Math.max(0,Math.min(1,raw))*determinant;
    return [density,density*fixture.transportedVelocityFine[1]!,raw];
  };
}
