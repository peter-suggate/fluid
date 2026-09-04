import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12CellAccessWGSL,
  createSparseCM12RowAccessWGSL,
  SPARSE_CM12_ATOMIC_ARENA_READERS,
} from "../lib/methods/adaptive-mass/sparse-cm12-row-access.wgsl";

test("dynamic B8 cell geometry is derived instead of loaded from page records", () => {
  const wgsl = createSparseCM12CellAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, true);
  assert.match(wgsl, /fn dynamicCellMinimum\(local:u32\)->vec3i/);
  assert.match(wgsl,
    /cm12WorldLeafCoordinate\(dynamicCellLeaf\(local\)\)\*i32\(BRICK_FINE_RESOLUTION\)/);
  assert.match(wgsl, /if\(id>=host\)\{return 1\.0;\}/);
  assert.doesNotMatch(wgsl, /candidateTopologyPageBase\(page\)\+16u\+8u/);
  assert.doesNotMatch(wgsl, /fn cellBase\(/);
});

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

test("dynamic incidence and term records decode one address for both fields", () => {
  const wgsl = createSparseCM12RowAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, true);
  const range = wgsl.slice(wgsl.indexOf("fn incidenceRange"), wgsl.indexOf(
    "fn incidenceBegin", wgsl.indexOf("fn incidenceRange")));
  assert.equal((range.match(/local\/cells/g) ?? []).length, 1);
  assert.equal((range.match(/local%cells/g) ?? []).length, 1);
  assert.equal((range.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 0);
  assert.match(range, /let begin=ta\(5u\)\+page\*\(6u\*cells\)\+6u\*within/);
  assert.match(range, /return vec2u\(begin,begin\+6u\)/);

  const incidence = wgsl.slice(wgsl.indexOf("fn incidenceRecord"), wgsl.indexOf(
    "fn incidenceRow", wgsl.indexOf("fn incidenceRecord")));
  assert.equal((incidence.match(/local\/records/g) ?? []).length, 1);
  assert.equal((incidence.match(/local%records/g) ?? []).length, 1);
  assert.equal((incidence.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.match(incidence, /dynamicIncidenceOverrideAt\(pageBase,cellWithin,side\)/);
  assert.match(incidence, /return vec2u\(ta\(at\),ta\(at\+1u\)\)/);

  const term = wgsl.slice(wgsl.indexOf("fn termRecord"), wgsl.indexOf(
    "fn termCell", wgsl.indexOf("fn termRecord")));
  assert.equal((term.match(/local\/terms/g) ?? []).length, 1);
  assert.equal((term.match(/local%terms/g) ?? []).length, 1);
  assert.equal((term.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.match(term, /let at=base\+ta\(base\+8u\)\+2u\*within/);
  assert.match(term, /return vec2u\(ta\(at\),ta\(at\+1u\)\)/);
});
