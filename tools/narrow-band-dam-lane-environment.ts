/** The retained solver's dam scenes for Instruments captures. */
export const NARROW_BAND_DAM_LANES = {
  mini: { FLUID_SCENE: "minimal-power-dam-break-64", FLUID_EXPECT_GRID: "64,64,64", FLUID_ORACLE_STEPS: "120" },
  large: { FLUID_SCENE: "large-power-dam-break", FLUID_EXPECT_GRID: "64,20,64", FLUID_ORACLE_STEPS: "120" },
} as const;
export type NarrowBandDamLane = keyof typeof NARROW_BAND_DAM_LANES;

export function narrowBandDamLaneWithSteps(lane: NarrowBandDamLane, steps: number): Record<string, string> {
  const dt = 0.017;
  return {
    ...NARROW_BAND_DAM_LANES[lane],
    FLUID_METHOD: "uniform-narrow-band-flip",
    FLUID_MAX_DT: String(dt),
    FLUID_TARGET_S: String(steps * dt),
    FLUID_ORACLE_STEPS: String(steps),
  };
}
