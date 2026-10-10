import { create } from "zustand";
import type { GPUQuality } from "../gpu-quality";
import { defaultMethodId, getMethod } from "../method-registry";
import { resolveMethodValues, type MethodParamValue, type MethodParamValues, type MethodProfile } from "../method-contract";

/**
 * The installed solver configuration. The registry key keeps this store
 * compatible with plugin consumers; only the application default is admitted.
 */
interface MethodStore {
  methodId: string;
  quality: GPUQuality;
  overrides: Record<string, MethodParamValues>;
  setQuality: (quality: GPUQuality) => void;
  setParam: (methodId: string, key: string, value: MethodParamValue) => void;
  resetParam: (methodId: string, key: string) => void;
  resetParams: (methodId: string) => void;
  applyProfile: (profile: MethodProfile) => void;
  seedProfile: (profile: MethodProfile) => void;
}

/** Shared preflight for UI commands, profiles, and direct store callers. */
export function validateMethodConfiguration(methodId: string, quality: GPUQuality, overrides: MethodParamValues): void {
  const method = getMethod(methodId);
  if (method.id !== methodId) throw new Error(`Unknown simulation method ${methodId}`);
  resolveMethodValues(method, quality, overrides);
}

export const createMethodStore = () => create<MethodStore>((set) => ({
  methodId: defaultMethodId(),
  quality: "balanced",
  overrides: {},
  setQuality: (quality) => set(state => {
    validateMethodConfiguration(state.methodId, quality, state.overrides[state.methodId] ?? {});
    return { quality };
  }),
  setParam: (methodId, key, value) => set(state => {
    const overrides = { ...state.overrides[methodId], [key]: value };
    validateMethodConfiguration(methodId, state.quality, overrides);
    return { overrides: { ...state.overrides, [methodId]: overrides } };
  }),
  resetParam: (methodId, key) => set(state => {
    const rest = { ...(state.overrides[methodId] ?? {}) };
    delete rest[key];
    validateMethodConfiguration(methodId, state.quality, rest);
    return { overrides: { ...state.overrides, [methodId]: rest } };
  }),
  resetParams: methodId => set(state => {
    validateMethodConfiguration(methodId, state.quality, {});
    return { overrides: { ...state.overrides, [methodId]: {} } };
  }),
  applyProfile: (profile) => set(() => {
    const methodId = defaultMethodId();
    const overrides = profile.methodId === methodId ? profile.overrides : {};
    validateMethodConfiguration(methodId, profile.quality, overrides);
    return { methodId, quality: profile.quality, overrides: { [methodId]: { ...overrides } } };
  }),
  // Retired scene profiles contribute no settings to the retained solver.
  seedProfile: ({ methodId, overrides }) => set(state => {
    if (methodId !== defaultMethodId()) return state;
    validateMethodConfiguration(methodId, state.quality, overrides);
    return { overrides: { [methodId]: { ...overrides } } };
  }),
}));

export type MethodStoreHook = ReturnType<typeof createMethodStore>;

/**
 * The default (pane A) instance.
 *
 * Per-pane instances come from `createPaneSession`; this one is what a tree
 * with no `SessionProvider` mounted reads, and what non-React callers that
 * have not yet been threaded a session resolve to.
 */
// Production chunk evaluation can load a UI dependency before the entry
// point's method-catalog side effect. Resolve the default on first use, once
// the entry point has finished installing methods, rather than on import.
let defaultStore: MethodStoreHook | undefined;
const defaultMethodStore = () => defaultStore ??= createMethodStore();
export const useMethodStore = new Proxy(
  (...args: unknown[]) => Reflect.apply(defaultMethodStore(), undefined, args),
  { get: (_target, key) => Reflect.get(defaultMethodStore(), key) },
) as MethodStoreHook;

/** Effective values for the active method: defaults ← quality preset ← app defaults ← user overrides. */
export function resolvedMethodValues(state: Pick<MethodStore, "methodId" | "quality" | "overrides">): MethodParamValues {
  const method = getMethod(state.methodId);
  return resolveMethodValues(method, state.quality, { ...method.appDefaults, ...state.overrides[state.methodId] });
}
