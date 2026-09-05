import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AdaptiveMassToolstripRow } from "../components/AdaptiveMassToolstripRow";

test("default adaptive criterion has its own chevron and three visible primary inputs", () => {
  const html = renderToStaticMarkup(createElement(AdaptiveMassToolstripRow));
  assert.match(html, /data-testid="scene-adaptivity-row"/);
  assert.match(html, /data-testid="scene-adaptivity-pick"/);
  assert.match(html, /aria-haspopup="menu"/);
  assert.match(html, /Coarse first/);
  assert.equal((html.match(/type="number"/g) ?? []).length, 3);
  assert.match(html, /Finest kinetic energy/);
  assert.match(html, /Curvature tolerance/);
  assert.match(html, /Impact lookahead/);
});
