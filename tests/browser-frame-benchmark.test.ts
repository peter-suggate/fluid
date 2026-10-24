import assert from "node:assert/strict";
import test from "node:test";
import { createBrowserFrameBenchmark } from "../lib/core/browser-frame-benchmark";
import type { RendererFrameMetrics } from "../lib/core/webgpu-renderer";

test("complete-frame measurement discards startup, readiness losses, context changes and resizes", () => {
  const previous = new Map(["window", "document", "performance", "console"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const elements = new Map<string, FakeElement>();
  let now = 0;
  class FakeElement {
    id = ""; textContent = ""; title = ""; style = { cssText: "" }; onclick?: () => void;
    remove() { elements.delete(this.id); }
  }
  Object.defineProperties(globalThis, {
    window: { configurable: true, value: { location: { search: "?frameBenchmark=test" } } },
    document: { configurable: true, value: { createElement: () => new FakeElement(), getElementById: (id: string) => elements.get(id), body: { append: (e: FakeElement) => elements.set(e.id, e) } } },
    performance: { configurable: true, value: { now: () => now } },
    console: { configurable: true, value: { ...console, info: () => {} } },
  });
  try {
    const benchmark = createBrowserFrameBenchmark()!;
    const button = elements.get("browser-frame-benchmark-run")!;
    const metrics = { context: "scene:ao:water:shadows" } as RendererFrameMetrics;
    let completed = 0;
    const sample = (ready = true, resolution = "3474 × 1882", context = metrics.context) => {
      now += 8; benchmark.sample(++completed, .3, { ...metrics, context }, resolution, { scene: "test", camera: { distance_m: 2 } }, ready);
    };
    const output = () => elements.get("browser-frame-benchmark");
    for (const change of ["readiness", "resize", "context"]) {
      button.onclick!();
      for (let i = 0; i < 300; i++) sample(false);
      assert.equal(output(), undefined, "startup never contributes completed frames");
      for (let i = 0; i < 250; i++) sample();
      assert.equal(output(), undefined, "warm-up plus partial measurement is insufficient");
      let resolution = "3474 × 1882", context = metrics.context;
      if (change === "readiness") sample(false);
      if (change === "resize") { resolution = "1920 × 1080"; sample(true, resolution); }
      if (change === "context") { context = "changed:ao:water:shadows"; sample(true, resolution, context); }
      for (let i = 0; i < 480; i++) sample(true, resolution, context);
      assert.equal(output(), undefined, `${change} discards the entire previous window`);
      sample(true, resolution, context);
      const result = JSON.parse(output()!.textContent);
      assert.equal(result.completed, 360);
      assert.equal(result.elapsed_ms, 2880);
      assert.equal(result.frame_ms, 8);
      assert.equal(result.resolution, resolution);
      assert.equal(result.context, context);
    }
    benchmark.dispose();
    assert.equal(elements.size, 0);
  } finally {
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  }
});
