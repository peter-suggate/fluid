import type { EnvironmentId } from "../environments";
import type { EnvironmentSceneryModule } from "./builder";
import { bathhouseScenery } from "./bathhouse";
import { concreteGalleryScenery } from "./concrete-gallery";
import { conservatoryScenery } from "./conservatory";
import { courtyardScenery } from "./courtyard";
import { defaultStudioScenery } from "./default-studio";
import { gardenScenery } from "./garden";
import { nightLabScenery } from "./night-lab";
import { researchStationScenery } from "./research-station";

export * from "./builder";

/**
 * Every environment's geometry, one module each. A scene is re-art-directed by
 * editing its own file: nothing here is shared but the builder vocabulary, so
 * two environments can be worked on at once without touching the same source.
 */
export const ENVIRONMENT_SCENERY: Readonly<Record<EnvironmentId, EnvironmentSceneryModule>> = Object.freeze({
  conservatory: conservatoryScenery,
  courtyard: courtyardScenery,
  "night-lab": nightLabScenery,
  "concrete-gallery": concreteGalleryScenery,
  bathhouse: bathhouseScenery,
  "research-station": researchStationScenery,
  default: defaultStudioScenery,
  garden: gardenScenery,
});

export function environmentScenery(id: EnvironmentId): EnvironmentSceneryModule {
  const scenery = ENVIRONMENT_SCENERY[id];
  if (!scenery) throw new Error(`No scenery module for environment ${id}`);
  return scenery;
}
