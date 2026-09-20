import {
  createSceneQueryLayerCache,
  isSceneQueryKey,
  parseQueryState,
} from "../lib/core/url-state";
import type { PaneSession } from "../lib/core/session/session";
import { createStore } from "zustand/vanilla";
import {
  booleanQuery,
  choiceQuery,
  combineQueryCodecs,
  queryRecord,
} from "../lib/framework/persistence";
import { startHostQueryStateSync } from "../lib/core/query-state-sync";
import { labMethodFromSearch } from "./lab-method";
import { labSceneQuery } from "./lab-scenes";
import { labStepQuery } from "./lab-step";
import { sliceViewQuery, type SliceViewFraction } from "./view-transform";

export const UNIFORM_LENSES = [
  ["surface", "Surface · phi = 0"],
  ["volume", "Conserved volume"],
  ["pressure", "Pressure"],
  ["velocity", "Velocity"],
  ["tiles", "Work tiles"],
  ["release", "Released faces"],
] as const;
export type UniformLens = (typeof UNIFORM_LENSES)[number][0];
export interface UniformLabState {
  sceneId: string;
  dt: number;
  lens: UniformLens;
  grid: boolean;
  sliceView: SliceViewFraction;
}
const fields = queryRecord({
  lens: choiceQuery<UniformLens>(
    "field",
    "surface",
    UNIFORM_LENSES.map(([id]) => id),
  ),
  grid: booleanQuery("grid", false),
});
export const uniformLabQuery = combineQueryCodecs<UniformLabState>([
  labSceneQuery,
  labStepQuery,
  fields,
  sliceViewQuery,
]);
export function createUniformLabStore(search: string) {
  return createStore<UniformLabState>(() =>
    uniformLabQuery.read(new URLSearchParams(search)),
  );
}
export type UniformLabStore = ReturnType<typeof createUniformLabStore>;
export function startUniformLabQuerySync(
  store: UniformLabStore,
  session: PaneSession,
  beforeSceneChange: () => void,
) {
  const sceneEntries = createSceneQueryLayerCache();
  let hydrated = false;
  return startHostQueryStateSync({
    path: "/advance-lab",
    hydrate: (search) => {
      if (labMethodFromSearch(search) !== "uniform-volume") return;
      const next = uniformLabQuery.read(new URLSearchParams(search));
      const query = new URLSearchParams(search);
      query.set("scene", next.sceneId);
      const parsed = parseQueryState(query.toString());
      if (
        hydrated &&
        (next.sceneId !== store.getState().sceneId ||
          JSON.stringify(parsed.scene) !==
            JSON.stringify(session.scene.getState().scene))
      )
        beforeSceneChange();
      hydrated = true;
      session.scene.getState().setScene(parsed.scene, next.sceneId);
      store.setState(next);
    },
    serialize: (search) => {
      if (labMethodFromSearch(search) !== "uniform-volume")
        return new URLSearchParams(search).toString();
      const query = new URLSearchParams(search);
      for (const key of [...query.keys()])
        if (isSceneQueryKey(key)) query.delete(key);
      for (const [key, value] of sceneEntries(session.scene.getState()))
        query.set(key, value);
      uniformLabQuery.write(query, store.getState());
      query.set("method", "uniform-volume");
      return query.toString();
    },
    sources: [
      (onChange) => store.subscribe(onChange),
      (onChange) => session.scene.subscribe(onChange),
    ],
  });
}
