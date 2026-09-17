import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12CellAccessWGSL,
  createSparseCM12RowAccessWGSL,
  SPARSE_CM12_ATOMIC_ARENA_READERS,
} from "../lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl";


test("authored cells retain their packed eight-word geometry", () => {
  const wgsl = createSparseCM12CellAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, false);
  assert.match(wgsl, /fn cellBase\(id:u32\)->u32\{return ta\(6u\)\+id\*8u;\}/);
  assert.match(wgsl, /fn cellCenter\(id:u32\)->vec3f/);
  assert.match(wgsl, /fn cellWidths\(id:u32\)->vec3f/);
});

test("dynamic row centers decode their page address once", () => {
  const wgsl = createSparseCM12RowAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, true);
  const center = wgsl.slice(wgsl.indexOf("fn rowCenter"), wgsl.indexOf(
    "fn termRecord", wgsl.indexOf("fn rowCenter")));
  assert.equal((center.match(/local\/rows/g) ?? []).length, 1);
  assert.equal((center.match(/local%rows/g) ?? []).length, 1);
  assert.equal((center.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.doesNotMatch(center, /rowWord\(id,[678]u\)/);
  const dynamic = center.slice(center.indexOf("let rows="));
  assert.match(dynamic,
    /vec3f\(taf\(base\+4u\*rows\),taf\(base\+5u\*rows\),taf\(base\+6u\*rows\)\)/);
});

test("dynamic row semantic planes map to their compact stored order", () => {
  const wgsl = createSparseCM12RowAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, true);
  const rowWord = wgsl.slice(wgsl.indexOf("fn rowWord"),
    wgsl.indexOf("fn boundedIncidenceEnd"));
  assert.match(rowWord, /if\(plane==2u\)\{storedPlane=3u;\}/,
    "semantic dual weight must read compact plane 3");
  assert.match(rowWord, /if\(plane==4u\)\{storedPlane=2u;\}/,
    "semantic distance must read compact plane 2");
  assert.match(rowWord, /if\(plane>=6u\)\{storedPlane=plane-2u;\}/,
    "semantic centers must read compact planes 4 through 6");
  assert.doesNotMatch(rowWord, /storedPlane-=1u/);
});

test("static row centers retain direct plane addressing", () => {
  const wgsl = createSparseCM12RowAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, false);
  assert.match(wgsl,
    /fn rowCenter\(id:u32\)->vec3f\{return vec3f\(taf\(rowWord\(id,6u\)\),taf\(rowWord\(id,7u\)\),taf\(rowWord\(id,8u\)\)\);\}/);
});

