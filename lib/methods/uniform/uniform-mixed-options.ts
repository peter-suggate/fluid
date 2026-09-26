/** Initial mixed-frame contract. Keep optional numerical experiments out of
 * the live path until their owner/seam semantics have been verified. This is
 * checked before consuming fields, regardless of the current ownership layout.
 */
export interface UniformMixedOptionalFeatures {
  velocityTransport: string;
  airborneMomentum: boolean;
  liquidOnlyVelocityAdvection: boolean;
  volumePressureRows: number;
  volumeCompaction: boolean;
  phiSeedFromVolume: boolean;
  phiAgreementGain: number;
  redistanceSurface: string;
  orphanVolume: string;
  orphanVolumeRender: string;
  isolatedBodyVolume: boolean;
  phiSeedCells: boolean;
}

export function assertUniformMixedOptions(options: UniformMixedOptionalFeatures): void {
  const unsupported: string[] = [];
  if (options.velocityTransport !== "semi-lagrangian") unsupported.push("MacCormack momentum");
  if (options.airborneMomentum) unsupported.push("airborne momentum");
  if (options.liquidOnlyVelocityAdvection) unsupported.push("liquid-only momentum advection");
  if (options.volumePressureRows !== 0) unsupported.push("volume pressure rows");
  if (options.volumeCompaction) unsupported.push("volume compaction");
  if (options.phiSeedFromVolume) unsupported.push("seed phi from volume");
  if (options.phiAgreementGain !== 0) unsupported.push("phi follows volume");
  if (options.redistanceSurface !== "auto" && options.redistanceSurface !== "rebuild") unsupported.push("preserved/sparse surface redistancing");
  if (options.orphanVolume !== "relay") unsupported.push("local/compact orphan volume");
  if (options.orphanVolumeRender !== "off") unsupported.push("orphan volume rendering");
  if (options.isolatedBodyVolume) unsupported.push("isolated body volume");
  if (options.phiSeedCells) unsupported.push("seed phi from V cells");
  if (unsupported.length) throw new Error(`Mixed Uniform currently requires these options off or at their defaults: ${unsupported.join(", ")}. Restore the supported defaults to continue.`);
}
