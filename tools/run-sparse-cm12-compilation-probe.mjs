/** Explicit single-child instrumentation; never preload the canonical runner.
 *
 * env -u NODE_OPTIONS node tools/run-sparse-cm12-compilation-probe.mjs \
 *   -- tools/probe-sparse-cm12-stage-cost.ts --scene=... [ordinary probe args]
 *
 * Optional --node-arg=--max-old-space-size=8192 preserves an explicit child
 * runtime option. --probe=PATH supports a different diagnostic observer.
 * This is a diagnostic run, not a canonical performance acceptance.
 */
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const launcher = fileURLToPath(import.meta.url);
const childFlag = "--cm12-compilation-probe-child";
const defaultProbe = fileURLToPath(new URL("./probe-sparse-cm12-compilation.ts", import.meta.url));

export function compilationProbeChildPlan(argv, environment = process.env) {
  const divider = argv.indexOf("--");
  if (divider < 0 || divider === argv.length - 1) throw new Error("Expected -- ENTRY [entry arguments]");
  const nodeArgs = []; let probe = defaultProbe;
  for (const option of argv.slice(0, divider)) {
    if (option.startsWith("--node-arg=")) {
      const value = option.slice("--node-arg=".length);
      if (!value.startsWith("--") || /^(?:--import|--require|--loader|--experimental-loader)(?:=|$)/.test(value)) {
        throw new Error("Preloads are not child runtime options; use --probe=PATH");
      }
      nodeArgs.push(value);
    } else if (option.startsWith("--probe=") && option.length > 8) probe = resolve(option.slice(8));
    else throw new Error(`Unknown launcher option ${option}`);
  }
  const [entry, ...entryArgs] = argv.slice(divider + 1);
  const env = { ...environment }; delete env.NODE_OPTIONS;
  return { argv: [...nodeArgs, launcher, childFlag, probe, resolve(entry), ...entryArgs], env };
}

async function runChildEntry() {
  const [probe, entry, ...args] = process.argv.slice(3);
  if (!probe || !entry || process.env.NODE_OPTIONS) throw new Error("Invalid isolated probe child");
  process.argv = [process.execPath, entry, ...args];
  const report = phase => console.error(JSON.stringify({ probe: "cm12-probe-launcher", phase,
    pid: process.pid, entry, observer: probe, instrumented: true, canonicalTiming: false }));
  report("bootstrap");
  // Deliberately dynamic and sequential. The JS entry starts without any TS
  // preload in execArgv/NODE_OPTIONS; tsx installs its loaders first. Loading
  // the observer as an ordinary module does not preload it into its workers.
  await import("tsx");
  report("typescript-ready");
  await import(pathToFileURL(probe).href);
  report("observer-ready");
  await import(pathToFileURL(entry).href);
}

async function launch() {
  const plan = compilationProbeChildPlan(process.argv.slice(2));
  const child = spawn(process.execPath, plan.argv, { env: plan.env, stdio: "inherit" });
  const signals = ["SIGINT", "SIGTERM"];
  const handlers = signals.map(signal => { const handler = () => child.kill(signal); process.on(signal, handler); return handler; });
  try {
    const result = await new Promise((resolveExit, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => resolveExit({ code, signal }));
    });
    for (let i = 0; i < signals.length; i++) process.removeListener(signals[i], handlers[i]);
    if (result.signal) process.kill(process.pid, result.signal);
    else process.exitCode = result.code ?? 1;
  } finally {
    for (let i = 0; i < signals.length; i++) process.removeListener(signals[i], handlers[i]);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === launcher) {
  try {
    if (process.argv[2] === childFlag) await runChildEntry();
    else await launch();
  } catch (error) {
    console.error(error instanceof Error ? error.stack : error);
    process.exitCode = 1;
  }
}
