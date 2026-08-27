import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12RowAccessWGSL,
  SPARSE_CM12_ATOMIC_ARENA_READERS,
} from "../lib/methods/adaptive-mass/sparse-cm12-row-access.wgsl";

test("dynamic row centers decode their page address once", () => {
  const wgsl = createSparseCM12RowAccessWGSL(
    SPARSE_CM12_ATOMIC_ARENA_READERS, true);
  const center = wgsl.slice(wgsl.indexOf("fn rowCenter"), wgsl.indexOf(
    "fn termRecord", wgsl.indexOf("fn rowCenter")));
  assert.equal((center.match(/local\/rows/g) ?? []).length, 1);
  assert.equal((center.match(/local%rows/g) ?? []).length, 1);
  assert.equal((center.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.doesNotMatch(center, /rowWord\(id,[678]u\)/);
  assert.match(center,
    /vec3f\(taf\(base\+6u\*rows\),taf\(base\+7u\*rows\),taf\(base\+8u\*rows\)\)/);
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
  assert.equal((range.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.match(range, /return vec2u\(begin,boundedIncidenceEnd\(begin,end\)\)/);

  const incidence = wgsl.slice(wgsl.indexOf("fn incidenceRecord"), wgsl.indexOf(
    "fn incidenceRow", wgsl.indexOf("fn incidenceRecord")));
  assert.equal((incidence.match(/local\/records/g) ?? []).length, 1);
  assert.equal((incidence.match(/local%records/g) ?? []).length, 1);
  assert.equal((incidence.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.match(incidence, /return vec2u\(ta\(at\),ta\(at\+1u\)\)/);

  const term = wgsl.slice(wgsl.indexOf("fn termRecord"), wgsl.indexOf(
    "fn termCell", wgsl.indexOf("fn termRecord")));
  assert.equal((term.match(/local\/terms/g) ?? []).length, 1);
  assert.equal((term.match(/local%terms/g) ?? []).length, 1);
  assert.equal((term.match(/candidateTopologyPageBase\(page\)/g) ?? []).length, 1);
  assert.match(term, /return vec2u\(ta\(at\),ta\(at\+1u\)\)/);
});
