import { createStore, type StoreApi } from "zustand/vanilla";
import type { WidgetPlacement, WidgetPlacementOverrides } from "../../framework/widgets";

/**
 * The user's own widget arrangement, per pane and per scene, kept in local
 * storage. Storage can be denied or cleared; the layout then falls back to
 * the scene's and nothing else changes.
 */

const STORAGE_KEY = "fluid-lab-control-widgets";

type Layouts = Readonly<Record<string, WidgetPlacementOverrides>>;

export interface WidgetLayoutState {
  readonly layouts: Layouts;
  place(sceneId: string, widgetId: string, placement: Partial<WidgetPlacement>): void;
  /** One widget back to the scene's placement, or every widget when none is named. */
  reset(sceneId: string, widgetId?: string): void;
}

function storageKey(paneId: string) { return `${STORAGE_KEY}:${paneId}`; }

function read(paneId: string): Layouts {
  try {
    const raw = window.localStorage.getItem(storageKey(paneId));
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Layouts : {};
  } catch { return {}; }
}

function write(paneId: string, layouts: Layouts) {
  try { window.localStorage.setItem(storageKey(paneId), JSON.stringify(layouts)); } catch { /* the layout still holds for this visit */ }
}

const stores = new Map<string, StoreApi<WidgetLayoutState>>();

export function widgetLayoutStore(paneId: string): StoreApi<WidgetLayoutState> {
  const existing = stores.get(paneId);
  if (existing) return existing;
  const store = createStore<WidgetLayoutState>((set, get) => {
    const commit = (layouts: Layouts) => { write(paneId, layouts); set({ layouts }); };
    return {
      layouts: typeof window === "undefined" ? {} : read(paneId),
      place: (sceneId, widgetId, placement) => {
        const layouts = get().layouts;
        commit({ ...layouts, [sceneId]: { ...layouts[sceneId], [widgetId]: placement } });
      },
      reset: (sceneId, widgetId) => {
        const { [sceneId]: scene, ...rest } = get().layouts;
        if (!scene) return;
        if (widgetId === undefined) { commit(rest); return; }
        const { [widgetId]: _dropped, ...kept } = scene;
        commit(Object.keys(kept).length ? { ...rest, [sceneId]: kept } : rest);
      },
    };
  });
  stores.set(paneId, store);
  return store;
}
