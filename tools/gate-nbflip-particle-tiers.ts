/** Same-process identity gate for tools/probe-nbflip-particle-tiers.ts:
 * runs the probe from each tree in turn, in this one process, so the arms
 * share the process (separate processes do not repeat bit for bit).
 * node --import tsx tools/gate-nbflip-particle-tiers.ts ROOT=NAME... -- PROBE ARGS
 * Each ROOT is a tree holding lib/ and this probe; give the baseline twice,
 * from two copies, for the A/A control (one copy twice would share modules). */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const split = process.argv.indexOf("--");
assert.ok(split > 2, "usage: gate-nbflip-particle-tiers.ts ROOT=NAME... -- PROBE ARGS");
const arms = process.argv.slice(2, split).map(a => { const [root, name] = a.split("="); assert.ok(root && name, a); return { root: resolve(root), name }; });
const args = process.argv.slice(split + 1), node = process.argv.slice(0, 2);
for (const arm of arms) {
  console.log(`== ${arm.name} ${arm.root}`);
  process.argv = [...node, arm.name, ...args];
  await import(pathToFileURL(`${arm.root}/tools/probe-nbflip-particle-tiers.ts`).href);
}
