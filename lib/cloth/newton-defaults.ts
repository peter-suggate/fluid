import type { ClothNewtonSettings } from './types';

/** Newton-PCG caps and tolerances of a document that names none: the Bridson scenes' (CLOTH.newton*, C1 RESULT 21). */
export const CLOTH_NEWTON_SCENE_DEFAULTS: ClothNewtonSettings = Object.freeze({ iterations: 8, pcgIterations: 200, pcgTolerance: 1e-4, residualTolerance: 1e-5 });
