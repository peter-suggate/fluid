/** Mixed-frame contract: momentum options the mixed owners do not carry yet.
 * Checked before consuming fields, regardless of the current ownership layout.
 */
export interface UniformMixedOptionalFeatures {
  velocityTransport: string;
  liquidOnlyVelocityAdvection: boolean;
}

export function assertUniformMixedOptions(options: UniformMixedOptionalFeatures): void {
  const unsupported: string[] = [];
  if (options.velocityTransport !== "semi-lagrangian") unsupported.push("MacCormack momentum");
  if (options.liquidOnlyVelocityAdvection) unsupported.push("liquid-only momentum advection");
  if (unsupported.length) throw new Error(`Mixed Uniform currently requires these options off or at their defaults: ${unsupported.join(", ")}. Restore the supported defaults to continue.`);
}
