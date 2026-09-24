import { readVisualLayers, writeVisualLayers, legacyVisualLayers, type VisualLayerState } from "../lib/core/visual-layers";
import {
  createSceneQueryLayerCache,
  isSceneQueryKey,
  parseQueryState,
} from "../lib/core/url-state";
import type { PaneSession } from "../lib/core/session/session";
import { createStore } from "zustand/vanilla";
import { combineQueryCodecs } from "../lib/framework/persistence";
import { startHostQueryStateSync } from "../lib/core/query-state-sync";
import { labMethodFromSearch } from "./lab-method";
import { labSceneQuery } from "./lab-scenes";
import { labStepQuery } from "./lab-step";
import { sliceViewQuery, type SliceViewFraction } from "./view-transform";

export interface UniformLabState {
  sceneId: string;
  totalSurfaceVolume: boolean;
  surfaceDeficitBalancing: boolean;
  phiCubicAdvection: boolean;
  phiDrain: boolean;
  airborneMomentum: boolean;
  isolatedBodyVolume: boolean;
  phiSeedCells: boolean;
  dt: number;
  layers: VisualLayerState;
  sliceView: SliceViewFraction;
  sliceDepth_m?: number;
}
/** The 3D default is on; only a viewer's opt-out reaches the URL. */
type Toggle = "totalSurfaceVolume" | "surfaceDeficitBalancing" | "phiCubicAdvection" | "phiDrain" | "airborneMomentum" | "isolatedBodyVolume" | "phiSeedCells";
function toggleQuery(key: Toggle, defaultValue: boolean) {
  return {
    keys: [key],
    read: (query: URLSearchParams) => ({ [key]: query.get(key) === null ? defaultValue : query.get(key) === "1" }),
    write: (query: URLSearchParams, state: UniformLabState) => {
      if (state[key] === defaultValue) query.delete(key);
      else query.set(key, state[key] ? "1" : "0");
    },
  };
}
export const uniformLabQuery = combineQueryCodecs<UniformLabState>([
  labSceneQuery,
  labStepQuery,
  {
    keys: ["sliceDepth"],
    read: (query: URLSearchParams) => {
      const raw = query.get("sliceDepth");
      const value = raw === null || raw.trim() === "" ? undefined : Number(raw);
      return { sliceDepth_m: value !== undefined && Number.isFinite(value) ? value : undefined };
    },
    write: (query: URLSearchParams, state: UniformLabState) => {
      if (state.sliceDepth_m === undefined) query.delete("sliceDepth");
      else query.set("sliceDepth", String(state.sliceDepth_m));
    },
  },
  toggleQuery("totalSurfaceVolume", true),
  toggleQuery("surfaceDeficitBalancing", true),
  toggleQuery("phiCubicAdvection", true),
  toggleQuery("phiDrain", true),
  toggleQuery("airborneMomentum", true),
  toggleQuery("isolatedBodyVolume", false),
  toggleQuery("phiSeedCells", false),
  {
    keys: ["layers", "field", "grid"],
    read: (query: URLSearchParams) => ({
      layers: readVisualLayers(query.get("layers"), legacyVisualLayers(
        query.get("field") ?? "surface", query.get("grid") === "1",
      )),
    }),
    write: (query: URLSearchParams, state: UniformLabState) => {
      query.delete("field");
      query.delete("grid");
      query.set("layers", writeVisualLayers(state.layers));
    },
  },
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
          next.sliceDepth_m !== store.getState().sliceDepth_m ||
          next.totalSurfaceVolume !== store.getState().totalSurfaceVolume ||
          next.surfaceDeficitBalancing !== store.getState().surfaceDeficitBalancing ||
          next.phiCubicAdvection !== store.getState().phiCubicAdvection ||
          next.phiDrain !== store.getState().phiDrain ||
          next.airborneMomentum !== store.getState().airborneMomentum ||
          next.isolatedBodyVolume !== store.getState().isolatedBodyVolume ||
          next.phiSeedCells !== store.getState().phiSeedCells ||
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
