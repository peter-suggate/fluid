import assert from "node:assert/strict";
import test from "node:test";

import { sparseCM12HostTemplateVariantsEnabled } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("host rerung admission budgets expanded cells and rows, not only bricks", () => {
  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    1_152 * 64, 1_152 * 240, 1_152, 8,
  ), true, "the established long-dam compatibility envelope must remain admitted");

  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    1_900 * 64, 1_900 * 240, 1_900, 8,
  ), false,
  "a B4-heavy atlas must not hide its million-cell all-rung expansion behind 1,900 leaves");
});
