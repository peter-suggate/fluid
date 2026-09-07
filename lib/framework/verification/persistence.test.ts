import assert from "node:assert/strict";
import test from "node:test";
import { booleanQuery, choiceQuery, numberQuery, queryRecord, combineQueryCodecs } from "../persistence";
const codec = queryRecord({ enabled: booleanQuery("enabled", true), mode: choiceQuery("mode", "a", ["a", "b"]), size: numberQuery("size", 3, 1, 8) });
test("feature query codecs round-trip overrides and remove defaults without touching other owners", () => {
  const query = new URLSearchParams("external=keep&enabled=0&mode=b&size=8");
  assert.deepEqual(codec.read(query), { enabled: false, mode: "b", size: 8 });
  codec.write(query, { enabled: true, mode: "a", size: 3 });
  assert.equal(query.toString(), "external=keep");
  codec.write(query, { enabled: false, mode: "b", size: 6 });
  assert.deepEqual(codec.read(query), { enabled: false, mode: "b", size: 6 });
});
test("malformed query values fall back to each feature's defaults", () => {
  assert.deepEqual(codec.read(new URLSearchParams("enabled=no&mode=unknown&size=Infinity")), { enabled: true, mode: "a", size: 3 });
  for (const raw of ["", "0", "9", "NaN"]) assert.equal(codec.read(new URLSearchParams(`size=${raw}`)).size, 3);
});
test("composed writes are atomic and key ownership must be unique", () => {
  const extra = queryRecord({ limit: numberQuery("limit", 2, 1, 4) });
  const combined = combineQueryCodecs<{enabled:boolean;mode:"a"|"b";size:number;limit:number}>([codec, extra]);
  const query = new URLSearchParams("external=keep&mode=b");
  assert.throws(() => combined.write(query, { enabled: false, mode: "a", size: 4, limit: 9 }), /Invalid limit/);
  assert.equal(query.toString(), "external=keep&mode=b");
  assert.throws(() => combineQueryCodecs([codec, codec]), /same query key/);
});
