import assert from "node:assert/strict";
import test from "node:test";
import { parseTraceTable } from "../tools/xctrace-trace-tables";

test("parseTraceTable resolves whole-backtrace references", async () => {
  const xml = `<node><schema>
    <col><mnemonic>time</mnemonic></col>
    <col><mnemonic>stack</mnemonic></col>
  </schema>
  <row><sample-time id="1" fmt="00:01.000.000">1000000</sample-time>
    <tagged-backtrace id="2" fmt="leaf ← parent"><backtrace>
      <frame id="3" name="leaf"/><frame id="4" name="parent"/>
    </backtrace></tagged-backtrace></row>
  <row><sample-time id="5" fmt="00:01.001.000">1001000</sample-time>
    <tagged-backtrace ref="2"/></row></node>`;
  const source = (async function* () { yield xml; })();
  const rows = [];
  for await (const row of parseTraceTable(source, { stacks: true })) rows.push(row);

  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].frames, ["leaf", "parent"]);
  assert.deepEqual(rows[1].frames, ["leaf", "parent"]);
  assert.equal(rows[1].stack, "leaf ← parent");
});
