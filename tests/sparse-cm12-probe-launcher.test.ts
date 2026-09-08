import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { observeMethodCalls } from "../tools/sparse-cm12-call-observer";
import { compilationProbeChildPlan } from "../tools/run-sparse-cm12-compilation-probe.mjs";

test("call observers preserve every argument, receiver, returned promise and thrown object", async () => {
  const token = {}, reporter = () => {}, options = Object.freeze({ enabled: false });
  const geometry = { get cells() { throw new Error("observer must not materialize geometry"); } };
  const args = [token, geometry, false, undefined, options, reporter];
  const receiver = {}, result = Promise.resolve(token), events: unknown[] = [];
  const methods = { configure(this: unknown, ...actual: unknown[]) {
    assert.equal(this, receiver); assert.equal(actual.length, args.length);
    for (let i = 0; i < args.length; i++) assert.equal(actual[i], args[i]);
    return result;
  } };
  const original = methods.configure;
  const restore = observeMethodCalls(methods, "configure", (phase, outcome) => events.push([phase, outcome]));
  assert.equal(Reflect.apply(methods.configure, receiver, args), result);
  assert.equal(await result, token); await Promise.resolve();
  assert.deepEqual(events, [["begin", undefined], ["end", "returned"]]);
  restore(); assert.equal(methods.configure, original);

  const error = new Error("original failure"), rejected = Promise.reject(error);
  const failures = { sync() { throw error; }, async() { return rejected; }, value() { return token; } };
  for (const name of ["sync", "async", "value"]) observeMethodCalls(failures, name, () => { throw new Error("observer failure"); });
  assert.throws(() => failures.sync(), value => value === error);
  assert.equal(failures.async(), rejected); await assert.rejects(rejected, value => value === error);
  assert.equal(failures.value(), token);
});

test("child plan isolates NODE_OPTIONS and preserves explicit runtime and entry arguments", () => {
  const environment = { NODE_OPTIONS: "--import tsx --import /bad/probe.ts", KEEP_ME: "unchanged" };
  const arguments_ = ["--node-arg=--max-old-space-size=4096", "--", "tools/probe-sparse-cm12-stage-cost.ts",
    "--scene=some scene", "--frames=2", "literal '$()'", "--import=ordinary-application-argument"];
  const plan = compilationProbeChildPlan(arguments_, environment);
  assert.equal(plan.env.NODE_OPTIONS, undefined); assert.equal(plan.env.KEEP_ME, "unchanged");
  assert.equal(environment.NODE_OPTIONS, "--import tsx --import /bad/probe.ts");
  assert.equal(plan.argv[0], "--max-old-space-size=4096");
  assert.deepEqual(plan.argv.slice(-4), arguments_.slice(-4));
  assert.ok(plan.argv.every((arg: string) => arg !== "npm" && arg !== "--import"));
  assert.throws(() => compilationProbeChildPlan(["--node-arg=--import=bad", "--", "x.ts"]), /Preloads/);
});

test("plain JS launcher loads its TypeScript observer only in the direct target child", { timeout: 15000 }, async () => {
  const directory = mkdtempSync(join(tmpdir(), "fluid-cm12-probe-"));
  const observer = join(directory, "observer.ts"), entry = join(directory, "entry.ts"), log = join(directory, "loads.jsonl");
  try {
    writeFileSync(observer, `import { appendFileSync } from "node:fs";
import { threadId } from "node:worker_threads";
const target = globalThis as typeof globalThis & { cm12ProbeCount?: number };
target.cm12ProbeCount = (target.cm12ProbeCount ?? 0) + 1;
appendFileSync(process.env.CM12_PROBE_TEST_LOG!, JSON.stringify({pid:process.pid,threadId})+"\\n");
`);
    writeFileSync(entry, `import { Worker } from "node:worker_threads";
import { execFileSync } from "node:child_process";
async function main() {
 const worker = new Worker('const {parentPort}=require("node:worker_threads");parentPort.postMessage({count:globalThis.cm12ProbeCount??0,nodeOptions:process.env.NODE_OPTIONS??null});', {eval:true});
 const workerResult = await new Promise(resolve => worker.once("message",resolve));
 await worker.terminate();
 const fork = JSON.parse(execFileSync(process.execPath,["-e",'console.log(JSON.stringify({count:globalThis.cm12ProbeCount??0,nodeOptions:process.env.NODE_OPTIONS??null}))'],{encoding:"utf8"}));
 console.log(JSON.stringify({pid:process.pid,args:process.argv.slice(2),count:(globalThis as any).cm12ProbeCount,worker:workerResult,fork}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
`);
    const env = { ...process.env, CM12_PROBE_TEST_LOG: log }; delete env.NODE_OPTIONS;
    const args = ["alpha beta", "literal '$()'", "--frames=2"];
    const output = execFileSync(process.execPath, [fileURLToPath(new URL("../tools/run-sparse-cm12-compilation-probe.mjs", import.meta.url)),
      `--probe=${observer}`, "--", entry, ...args], { env, encoding: "utf8", timeout: 10000,
      stdio: ["ignore", "pipe", "pipe"] });
    const receipt = JSON.parse(output.trim());
    assert.deepEqual(receipt.args, args); assert.equal(receipt.count, 1);
    assert.deepEqual(receipt.worker, { count: 0, nodeOptions: null });
    assert.deepEqual(receipt.fork, { count: 0, nodeOptions: null });
    assert.deepEqual(readFileSync(log, "utf8").trim().split("\n").map(line => JSON.parse(line)),
      [{ pid: receipt.pid, threadId: 0 }]);
    assert.notEqual(receipt.pid, process.pid);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
