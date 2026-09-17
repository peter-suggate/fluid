import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";

import { defaultMethodId, registeredSimulationMethods } from "../lib/core/method-registry";
import { useUIStore } from "../lib/core/stores/ui-store";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";

/**
 * Which keys the studio's canonical write claims — enumerated, never restated.
 *
 * A canonical write clears every key it owns and writes the current answer
 * back, so "is this key mine?" is the difference between *my state, stale* and
 * *somebody else's key* — an analytics tag, a router parameter, a link a reader
 * hand-edited. Since WP9 that predicate is composed from its owners
 * (`managedQueryKey` in `lib/framework/persistence.ts`) rather than written out
 * as one literal, and the thing worth pinning is that the composition claims
 * exactly what the literal did.
 *
 * The probe is behavioural and needs no export: put a sentinel on a key, write
 * a *default* state over it, and see whether the sentinel survives. A managed
 * key is deleted (and possibly rewritten); an unmanaged one is left alone.
 * Nothing here reads the module's source, and the positives are enumerated from
 * a fabricated maximal state, so a key some future feature adds arrives in this
 * test by itself.
 */

const base = parseQueryState("");
const defaults = {
  sceneState: { presetId: base.presetId, scene: base.scene },
  methodState: { methodId: base.methodId, quality: base.quality, overrides: base.overrides },
} as const;

/** True when a canonical write over `key` treats it as the studio's own. */
function managed(key: string): boolean {
  const search = new URLSearchParams([[key, "SENTINEL"]]).toString();
  const out = new URLSearchParams(serializeQueryState(search,
    defaults.sceneState as never, defaults.methodState as never));
  return out.get(key) !== "SENTINEL";
}

/** Every key a studio state that has touched everything puts in the address. */
function maximalKeys(): readonly string[] {
  const scene = JSON.parse(JSON.stringify(base.scene));
  scene.container.width_m = base.scene.container.width_m + 0.37;
  scene.fluid.density_kg_m3 = 1234;
  scene.fluid.gravity_m_s2 = { x: 0.5, y: -9.1, z: 0.25 };
  scene.numerics = { ...(scene.numerics ?? {}), pressureMaxIterations: 77 };
  scene.randomSeed = 4242;
  scene.fluid.refinementRegions = [{
    id: "region-1", rule: "minimum-cell-size", minimumCellSize_cells: 8,
    min_m: { x: -0.2, y: 0.05, z: -0.2 }, max_m: { x: 0.2, y: 0.4, z: 0.2 },
  }];
  scene.fluid.initialBrickSeeds_m = [{ x: 0.1, y: 0.2, z: 0.3 }];
  scene.fluid.initialBrickSeedsAdditive = true;

  const methods = registeredSimulationMethods();
  const overrides: Record<string, Record<string, number | string>> = {};
  for (const method of methods) {
    const values: Record<string, number | string> = {};
    for (const spec of method.params) {
      values[spec.key] = spec.kind === "select"
        ? spec.options[spec.options.length - 1]!.value
        : Math.min(spec.max, Math.max(spec.min, (spec.min + spec.max) / 2));
    }
    if (Object.keys(values).length > 0) overrides[method.id] = values;
  }
  const otherMethod = methods.find((m) => m.id !== defaultMethodId())?.id ?? defaultMethodId();

  const initialUI = useUIStore.getInitialState();
  const uiState = {
    ...initialUI,
    camera: {
      azimuth_rad: 0.77, elevation_rad: 0.41, distance_m: 3.3,
      tanHalfFov: initialUI.camera.tanHalfFov,
      target_m: { x: 0.11, y: 0.22, z: 0.33 },
    },
    sceneOverlay: "sim-pipeline",
    gridOverlayAxis: "y",
    gridOverlaySlice: 0.25,
    gridOverlayMode: "structure",
    gridOverlayLensPhase: 3,
  };
  const shellState = {
    view: "library",
    compare: {
      active: true,
      diff: { method: "uniform", gridMode: "phi", "scene.container.width_m": "2" },
      links: { view: true, cut: false, instrument: true, look: false, topology: true, regions: true },
      focusedPane: "a",
    },
  };

  const canonical = serializeQueryState("",
    { presetId: base.presetId, scene } as never,
    { methodId: otherMethod, quality: "ultra", overrides } as never,
    uiState as never, shellState as never, undefined, { topologyFrozen: true } as never);
  return [...new Set(new URLSearchParams(canonical).keys())];
}

test("every key the studio writes, it also claims", () => {
  const keys = maximalKeys();
  // A maximal state is broad, or this test proves nothing: the scene layer, the
  // method layer, the camera family, the compare diff and the feature codecs.
  assert.ok(keys.length >= 40, `only ${keys.length} keys emitted`);
  for (const family of ["camera.", "param.", "scene.", "b."]) {
    assert.ok(keys.some((key) => key.startsWith(family)), `no ${family}* key emitted`);
  }
  const unclaimed = keys.filter((key) => !managed(key));
  assert.deepEqual(unclaimed, [],
    `the writer emits these and would not clear them: ${unclaimed.join(", ")}`);
});

test("the retired names are still cleared, so an old link cannot outlive them", () => {
  // Nothing writes these any more. If the predicate stopped claiming them, a
  // link carrying one would keep it forever — the failure mode with no compile
  // error that `managedQueryKey`'s doc comment names.
  for (const key of ["panel", "panelWidth", "sceneConfig"]) {
    assert.equal(managed(key), true, key);
  }
});

test("a key that is not the studio's survives a canonical write", () => {
  // Two kinds of foreigner: somebody else's tag, and the 2-D lab's own keys —
  // which share the address space but never the address, and which the studio
  // must not silently delete if one is ever pasted onto a /scene link.
  for (const key of [
    "utm_source", "ref", "fbclid", "zzz",
    "transport", "solve", "surface", "overlays", "view.zoom", "view.x", "view.y",
    "cameraX", "paramX", "sceneX",
  ]) {
    assert.equal(managed(key), false, key);
  }
});

test("the keys both hosts share are the studio's on a /scene link", () => {
  // `scene`, `regions` and `gridMode` are declared once and mounted by both
  // pages. On this host they are the studio's to rewrite; §9.3's table is that
  // sharing, and this is the half of it a Node test can see.
  for (const key of ["scene", "regions", "gridMode"]) {
    assert.equal(managed(key), true, key);
  }
});
