import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { create } from "zustand";
import { resolveWaterOptics, resolveWaterKeyLight, packWaterSceneOptics, WATER_SCENE_OPTICS_BYTES, WATER_OPTICS } from "../lib/core/webgpu-lighting";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import { validateScene } from "../lib/core/model";
import { sceneEditRequiresReset } from "../lib/core/webgpu-renderer";
import { createPaneSession } from "../lib/core/session/session";
import { SessionProvider } from "../lib/core/session/session-context";
import { SurfaceDisplayRow } from "../lib/features/surface-display/ui";

test("water roughness reaches the GPU without changing physical Fresnel or receiver lanes", () => {
  const defaults = resolveWaterOptics();
  assert.equal(defaults.roughness, 0.12);
  assert.equal(defaults.fresnelF0, WATER_OPTICS.fresnelF0);
  assert.equal(resolveWaterOptics({ roughness: NaN }).roughness, defaults.roughness);
  assert.equal(resolveWaterOptics({ roughness: 0 }).roughness, 0.04);
  assert.equal(resolveWaterOptics({ roughness: 2 }).roughness, 1);
  const packed = packWaterSceneOptics(resolveWaterOptics({ roughness: 0.25 }), resolveWaterKeyLight());
  assert.equal(packed.byteLength, WATER_SCENE_OPTICS_BYTES);
  assert.equal(packed[28], 0.25);
  assert.equal(packed[7], Math.fround(WATER_OPTICS.fresnelF0));
  assert.deepEqual([...packed.slice(16, 24)], Array(8).fill(0));
});

test("roughness edits survive scene links, preserve other optics, and do not reset fluid", () => {
  const base = parseQueryState("?scene=water-box-dam-break");
  const optics = { ...base.scene.fluid.optics, roughness: 0.19, causticStrength: 0.25 };
  const scene = { ...base.scene, fluid: { ...base.scene.fluid, optics } };
  assert.deepEqual(validateScene(scene), []);
  assert.equal(sceneEditRequiresReset(base.scene, scene, base.methodId), false);
  const query = serializeQueryState("", { ...base, scene }, base, base.ui);
  assert.deepEqual(parseQueryState(query).scene.fluid.optics, optics);
  for (const roughness of [-1, 0, 1.1, Infinity, NaN]) {
    assert.ok(validateScene({ ...scene, fluid: { ...scene.fluid, optics: { roughness } } })
      .some(error => error.includes("Water roughness")));
  }
});

test("water roughness is exposed for shaded water, using the active scene value", () => {
  const session = createPaneSession("a");
  const draw = (mode: "shaded" | "simple" | "wireframe") => {
    // SSR reads the store's initial snapshot, so construct that snapshot with
    // the requested document and mode rather than mutating it after creation.
    const context = { ...session,
      scene: create(() => ({ ...session.scene.getState(), scene: { ...session.scene.getState().scene,
        fluid: { ...session.scene.getState().scene.fluid, optics: { roughness: 0.23 } } } })),
      ui: create(() => ({ ...session.ui.getState(), fluidSurfaceRenderMode: mode })),
    };
    return renderToStaticMarkup(<SessionProvider value={context}><SurfaceDisplayRow /></SessionProvider>);
  };
  assert.match(draw("shaded"), /aria-label="Water roughness"/);
  assert.match(draw("shaded"), /value="0.23"/);
  for (const mode of ["simple", "wireframe"] as const) {
    assert.doesNotMatch(draw(mode), /aria-label="Water roughness"/);
  }
});
