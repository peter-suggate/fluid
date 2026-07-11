export interface DamBreakFractions {
  width: number;
  height: number;
  depth: number;
}

const DAM_BREAK_HEIGHT = 0.92;

/**
 * Builds a square-footprint reservoir in the tank's lower (-x, -z) corner.
 * The footprint grows with requested fill so width * height * depth remains
 * equal to the scene fill fraction (up to a completely full tank).
 */
export function damBreakFractions(fillFraction: number): DamBreakFractions {
  const fill = Math.max(0, Math.min(1, fillFraction));
  if (fill === 0) return { width: 0, height: 0, depth: 0 };
  const height = Math.max(DAM_BREAK_HEIGHT, fill);
  const footprint = Math.sqrt(fill / height);
  return { width: footprint, height, depth: footprint };
}
