import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  parseWebGPUSmokeTimeout,
  WEBGPU_EXCLUSIVE_LOCK,
  WEBGPU_SMOKE_KILL_REAP_MS,
  WEBGPU_SMOKE_TERMINATE_GRACE_MS,
} from "./webgpu-smoke-isolation";

const timeout_ms = parseWebGPUSmokeTimeout(process.env.FLUID_WEBGPU_SMOKE_TIMEOUT_MS);
const worker = fileURLToPath(new URL("./run-webgpu-smoke-isolated-worker.ts", import.meta.url));

console.error("SAFETY: close every browser WebGPU tab before this isolated Dawn smoke. Never run Dawn and browser GPU validation concurrently.");
const child = spawn(process.execPath, ["--import", "tsx", worker], {
  cwd: process.cwd(),
  env: process.env,
  stdio: "inherit",
});
console.log(JSON.stringify({
  phase: "webgpu-smoke-launch",
  pid: child.pid,
  timeout_ms,
  exclusiveLock: WEBGPU_EXCLUSIVE_LOCK,
  isolation: "child-process",
}));

let timedOut = false;
let terminateTimer: ReturnType<typeof setTimeout> | undefined;
let forcedExitTimer: ReturnType<typeof setTimeout> | undefined;
const timeoutTimer = setTimeout(() => {
  timedOut = true;
  console.error(`WebGPU smoke PID ${child.pid ?? "unknown"} exceeded ${timeout_ms} ms; sending SIGTERM`);
  child.kill("SIGTERM");
  terminateTimer = setTimeout(() => {
    console.error(`WebGPU smoke PID ${child.pid ?? "unknown"} did not exit after SIGTERM; sending SIGKILL`);
    child.kill("SIGKILL");
    forcedExitTimer = setTimeout(() => {
      console.error(`WebGPU smoke PID ${child.pid ?? "unknown"} was not reaped after SIGKILL; leaving ${WEBGPU_EXCLUSIVE_LOCK} as owner evidence`);
      child.unref();
      process.exit(124);
    }, WEBGPU_SMOKE_KILL_REAP_MS);
    forcedExitTimer.unref();
  }, WEBGPU_SMOKE_TERMINATE_GRACE_MS);
  terminateTimer.unref();
}, timeout_ms);

child.once("error", (error) => {
  clearTimeout(timeoutTimer);
  if (terminateTimer) clearTimeout(terminateTimer);
  if (forcedExitTimer) clearTimeout(forcedExitTimer);
  console.error(error);
  process.exitCode = 1;
});

child.once("exit", (code, signal) => {
  clearTimeout(timeoutTimer);
  if (terminateTimer) clearTimeout(terminateTimer);
  if (forcedExitTimer) clearTimeout(forcedExitTimer);
  if (timedOut) process.exitCode = 124;
  else if (signal) {
    console.error(`WebGPU smoke PID ${child.pid ?? "unknown"} exited from signal ${signal}`);
    process.exitCode = 1;
  } else process.exitCode = code ?? 1;
});
