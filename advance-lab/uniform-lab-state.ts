import { readVisualLayers, writeVisualLayers, legacyVisualLayers, visualLayers, type VisualLayerState } from "../lib/core/visual-layers";
import {
  createSceneQueryLayerCache,
  isSceneQueryKey,
  parseQueryState,
} from "../lib/core/url-state";
import type { PaneSession } from "../lib/core/session/session";
import { createStore } from "zustand/vanilla";
import { combineQueryCodecs, numberQuery, queryRecord } from "../lib/framework/persistence";
import { NARROW_BAND_FINE_ADVECTION, NARROW_BAND_VOLUME_CONTROL } from "../lib/methods/uniform/uniform-narrow-band-controls";
import { UNIFORM_LAB_METHOD, type UniformLabControls } from "../lib/physics-wasm/uniform-controller";
import { startHostQueryStateSync } from "../lib/core/query-state-sync";
import { labMethodFromSearch } from "./lab-method";
import { labSceneQuery } from "./lab-scenes";
import { STEP_SIZES } from "./lab-step";
import { sliceViewQuery, type SliceViewFraction } from "./view-transform";

export interface UniformLabState extends UniformLabControls {
  sceneId: string;
  dt: number;
  layers: VisualLayerState;
  sliceView: SliceViewFraction;
  sliceDepth_m?: number;
}
/** The 3D method steps at 1/60 s by default; so does its lab. */
const stepQuery = queryRecord({
  dt: numberQuery("dt", 1 / 60, STEP_SIZES.at(-1)!.dt, STEP_SIZES[0].dt),
});
/** Defaults match the 3D narrow-band FLIP parameter schema. */
const controlQuery = combineQueryCodecs<UniformLabControls>([
  queryRecord({
    volumeControlSeconds: numberQuery(
      NARROW_BAND_VOLUME_CONTROL.key,
      NARROW_BAND_VOLUME_CONTROL.default,
      NARROW_BAND_VOLUME_CONTROL.min,
      NARROW_BAND_VOLUME_CONTROL.max,
    ),
  }),
  {
    keys: [NARROW_BAND_FINE_ADVECTION.key],
    read: (query: URLSearchParams) => ({
      phiFineAdvection: query.get(NARROW_BAND_FINE_ADVECTION.key) === "trilinear" ? "trilinear" as const : "cubic" as const,
    }),
    write: (query: URLSearchParams, state: UniformLabControls) => {
      if (state.phiFineAdvection === "cubic") query.delete(NARROW_BAND_FINE_ADVECTION.key);
      else query.set(NARROW_BAND_FINE_ADVECTION.key, state.phiFineAdvection);
    },
  },
]);
export const uniformLabQuery = combineQueryCodecs<UniformLabState>([
  labSceneQuery,
  stepQuery,
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
  controlQuery,
  {
    keys: ["layers", "field", "grid"],
    read: (query: URLSearchParams) => ({
      // The band's samples are this method's state: shown unless a link says otherwise.
      layers: readVisualLayers(query.get("layers"), query.has("field") || query.has("grid")
        ? legacyVisualLayers(query.get("field") ?? "surface", query.get("grid") === "1")
        : visualLayers(["surface", "particles"])),
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
      if (labMethodFromSearch(search) !== UNIFORM_LAB_METHOD) return;
      const next = uniformLabQuery.read(new URLSearchParams(search));
      const query = new URLSearchParams(search);
      query.set("scene", next.sceneId);
      const parsed = parseQueryState(query.toString());
      if (
        hydrated &&
        (next.sceneId !== store.getState().sceneId ||
          next.sliceDepth_m !== store.getState().sliceDepth_m ||
          next.volumeControlSeconds !== store.getState().volumeControlSeconds ||
          next.phiFineAdvection !== store.getState().phiFineAdvection ||
          JSON.stringify(parsed.scene) !==
            JSON.stringify(session.scene.getState().scene))
      )
        beforeSceneChange();
      hydrated = true;
      session.scene.getState().setScene(parsed.scene, next.sceneId);
      store.setState(next);
    },
    serialize: (search) => {
      if (labMethodFromSearch(search) !== UNIFORM_LAB_METHOD)
        return new URLSearchParams(search).toString();
      const query = new URLSearchParams(search);
      for (const key of [...query.keys()])
        if (isSceneQueryKey(key)) query.delete(key);
      for (const [key, value] of sceneEntries(session.scene.getState()))
        query.set(key, value);
      uniformLabQuery.write(query, store.getState());
      query.set("method", UNIFORM_LAB_METHOD);
      return query.toString();
    },
    sources: [
      (onChange) => store.subscribe(onChange),
      (onChange) => session.scene.subscribe(onChange),
    ],
  });
}
