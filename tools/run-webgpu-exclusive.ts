import { spawn } from "node:child_process";
import { constants } from "node:os";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const nodeArguments = process.argv.slice(2);
if (nodeArguments.length === 0) {
  throw new Error("run-webgpu-exclusive requires Node arguments for one GPU test or tool process");
}

const target = [process.execPath, ...nodeArguments].join(" ");
const evidence = await acquireWebGPUExclusiveLock("dawn-command", target);
console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "acquired", ...evidence }));

let child: ReturnType<typeof spawn> | undefined;
const forwardedSignals = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
const forward = (signal: typeof forwardedSignals[number]) => child?.kill(signal);
const handlers = new Map<typeof forwardedSignals[number], () => void>();
try {
  child = spawn(process.execPath, nodeArguments, {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });
  for (const signal of forwardedSignals) {
    const handler = () => forward(signal);
    handlers.set(signal, handler);
    process.once(signal, handler);
  }
  const outcome = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve, reject) => {
    child!.once("error", reject);
    child!.once("exit", (code, signal) => resolve({ code, signal }));
  });
  if (outcome.signal) {
    process.exitCode = 128 + (constants.signals[outcome.signal] ?? 0);
  } else {
    process.exitCode = outcome.code ?? 1;
  }
} finally {
  for (const [signal, handler] of handlers) process.removeListener(signal, handler);
  await releaseWebGPUExclusiveLock();
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "released", pid: process.pid }));
}
