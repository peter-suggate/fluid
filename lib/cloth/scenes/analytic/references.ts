/**
 * The closed-form and textbook answers the analytic cloth scenes are gated
 * against, in Float64. Nothing here runs the solver: these are the oracles.
 */

/**
 * A free particle under constant gravity through the solver's own time
 * integrator: one implicit Euler step (BDF2 has no history yet), then BDF2 at
 * the same step. BDF2 is exact for a quadratic trajectory, so after the start
 * the velocity is exactly g t and the position falls exactly g t^2 / 2 plus
 * the offset the Euler start leaves: e_n = (3/4) g h^2 (1 - 3^-n), which
 * settles at 3/4 g h^2 (2.04 mm at 1/60 s). Returns the displacement (along
 * gravity, negative down for g < 0) and velocity after each of `steps` steps.
 */
export function bdf2FreeFall(steps: number, h_s: number, g: number): { displacement_m: number[]; velocity_m_s: number[] } {
  const displacement_m: number[] = [], velocity_m_s: number[] = [];
  let x = 0, xPrev = 0, v = 0, vPrev = 0;
  for (let n = 0; n < steps; n++) {
    const bdf2 = n > 0, hEff = bdf2 ? 2 * h_s / 3 : h_s;
    const xHat = x + (bdf2 ? (x - xPrev) / 3 : 0), vHat = bdf2 ? (4 * v - vPrev) / 3 : v;
    const next = xHat + hEff * vHat + hEff * hEff * g;
    xPrev = x; x = next; vPrev = v; v = (next - xHat) / hEff;
    displacement_m.push(x); velocity_m_s.push(v);
  }
  return { displacement_m, velocity_m_s };
}

/** The Euler start's settled offset from g t^2 / 2: (3/4) g h^2. */
export function bdf2StartOffset_m(h_s: number, g: number): number {
  return 0.75 * g * h_s * h_s;
}

/**
 * Lane C1's exact equilibrium of a sheet hanging from its whole top edge
 * (tests/cloth-coarse.test.ts), Poisson 0 and area-lumped mass: band j (between
 * rows j and j + 1 of n rows at spacing h) carries the weight below its middle,
 * rho g (n - 1.5 - j) h per unit width, by uniaxial StVK k lambda (lambda^2 - 1) / 2.
 * Returns the band's engineering strain lambda - 1.
 */
export function hangingBandStrain(j: number, rows: number, h_m: number, k_N_m: number, rho_kg_m2: number, g: number): number {
  const load = rho_kg_m2 * g * (rows - 1.5 - j) * h_m;
  let l = 1;
  for (let it = 0; it < 60; it++) l -= (k_N_m * l * (l * l - 1) / 2 - load) / (k_N_m * (3 * l * l - 1) / 2);
  return l - 1;
}

/** The continuum hanging strain at depth s below the pinned edge of a sheet of length L: rho g (L - s) / k (small strain). */
export function hangingStrainContinuum(s_m: number, L_m: number, k_N_m: number, rho_kg_m2: number, g: number): number {
  return rho_kg_m2 * g * (L_m - s_m) / k_N_m;
}

/** Bending length c = (B / (rho g))^(1/3) (Peirce 1930). */
export function bendingLength_m(B_N_m: number, rho_kg_m2: number, g: number): number {
  return Math.cbrt(B_N_m / (rho_kg_m2 * g));
}

export interface HeavyElastica {
  /** Angle below the horizontal of the chord from the clamp to the tip (the D1388 line), degrees. */
  readonly chordAngle_deg: number;
  /** Tangent angle at the tip, degrees. */
  readonly tipSlope_deg: number;
  /** Tip below the clamp and beyond it, as fractions of the overhang length. */
  readonly tipDrop: number;
  readonly tipReach: number;
}

/**
 * The exact inextensible cantilever under its own weight (the heavy elastica),
 * clamped horizontal: with s the arc length over the overhang l and phi the
 * tangent angle below horizontal, phi'' = -lambda cos(phi) (1 - s), phi(0) = 0,
 * phi'(1) = 0, lambda = rho g l^3 / B = (l / c)^3. Shooting on phi'(0) with RK4.
 * At l / c = 2 it gives the 41.5 deg of ASTM D1388.
 */
export function heavyElastica(overhangOverBendingLength: number, steps = 4000): HeavyElastica {
  const lambda = overhangOverBendingLength ** 3;
  const shoot = (p: number) => {
    let phi = 0, dphi = p, X = 0, Y = 0;
    const ds = 1 / steps, f = (s: number, dp: number, ph: number) => [dp, -lambda * Math.cos(ph) * (1 - s)] as const;
    for (let i = 0; i < steps; i++) {
      const s = i * ds, c0 = Math.cos(phi), s0 = Math.sin(phi);
      const k1 = f(s, dphi, phi), k2 = f(s + ds / 2, dphi + ds / 2 * k1[1], phi + ds / 2 * k1[0]);
      const k3 = f(s + ds / 2, dphi + ds / 2 * k2[1], phi + ds / 2 * k2[0]), k4 = f(s + ds, dphi + ds * k3[1], phi + ds * k3[0]);
      phi += ds / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      dphi += ds / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      X += ds * (c0 + Math.cos(phi)) / 2; Y += ds * (s0 + Math.sin(phi)) / 2;
    }
    return { dphi, phi, X, Y };
  };
  let lo = 0, hi = lambda;
  for (let it = 0; it < 80; it++) { const mid = (lo + hi) / 2; if (shoot(mid).dphi > 0) hi = mid; else lo = mid; }
  const r = shoot((lo + hi) / 2), deg = 180 / Math.PI;
  return { chordAngle_deg: Math.atan2(r.Y, r.X) * deg, tipSlope_deg: r.phi * deg, tipDrop: r.Y, tipReach: r.X };
}

/** Peirce's approximation, (l / c)^3 = 8 tan(theta) / cos(theta / 2), solved for the chord angle theta (degrees). */
export function peirceChordAngle_deg(overhangOverBendingLength: number): number {
  const target = overhangOverBendingLength ** 3;
  let lo = 0, hi = Math.PI / 2 - 1e-9;
  for (let it = 0; it < 80; it++) { const t = (lo + hi) / 2; if (8 * Math.tan(t) / Math.cos(t / 2) > target) hi = t; else lo = t; }
  return lo * 180 / Math.PI;
}

/** A block on an incline with one Coulomb mu: 0 when it sticks (tan theta <= mu), else g (sin theta - mu cos theta). */
export function inclineAcceleration_m_s2(angle_deg: number, mu: number, g: number): number {
  const t = angle_deg * Math.PI / 180;
  return Math.tan(t) <= mu ? 0 : g * (Math.sin(t) - mu * Math.cos(t));
}
