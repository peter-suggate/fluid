/**
 * LANE F OWNS THIS FILE. It registers F's stages with the brick engine
 * (brick-stages.ts, brick-abi.md §6.2).
 *
 * Export the factory here when the particle, level-set and velocity stages
 * land. Until then it is undefined, and the method refuses every scene on the
 * brick engine (brick-method.ts). E's tests use bringUpStages instead.
 */
import type { BrickStagesFactory } from "./brick-stages";

export const brickFluidStages: BrickStagesFactory | undefined = undefined;
