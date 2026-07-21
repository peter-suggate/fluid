import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  parseWebGPUBringupStage,
  parseWebGPUBringupTimeout,
} from "./webgpu-bringup-stages";

const stage = parseWebGPUBringupStage(process.env.FLUID_BRINGUP_STAGE);
const timeout_ms = parseWebGPUBringupTimeout(process.env.FLUID_BRINGUP_TIMEOUT_MS);
const worker = fileURLToPath(new URL("./run-webgpu-bringup-stage-worker.ts", import.meta.url));

console.error("SAFETY: keep every browser WebGPU tab closed while this Dawn process runs. Never run Dawn and browser GPU validation concurrently.");
console.log(JSON.stringify({ phase: "bringup-launch", stage, timeout_ms, isolation: "child-process" }));

const child = spawn(process.execPath, ["--import", "tsx", worker], {
  cwd: process.cwd(),
  env: { ...process.env, FLUID_BRINGUP_STAGE: stage },
  stdio: "inherit",
});

let timedOut = false;
let forcedExitTimer: ReturnType<typeof setTimeout> | undefined;
const timer = setTimeout(() => {
  timedOut = true;
  console.error(`WebGPU bring-up stage ${stage} exceeded ${timeout_ms} ms; terminating its isolated child process`);
  child.kill("SIGTERM");
  setTimeout(() => {
    child.kill("SIGKILL");
    // A wedged Metal/Dawn process can remain in macOS's exiting state after
    // SIGKILL. Do not let that keep the launcher (and its caller) alive
    // indefinitely. The worker-owned exclusive lock intentionally remains,
    // preventing any later GPU run until the OS has actually reaped it.
    forcedExitTimer = setTimeout(() => {
      console.error(`WebGPU bring-up child for ${stage} did not exit after SIGKILL; leaving the exclusive GPU lock in place`);
      child.unref();
      process.exit(124);
    }, 2_000);
    forcedExitTimer.unref();
  }, 2_000).unref();
}, timeout_ms);

child.once("error", (error) => {
  clearTimeout(timer);
  console.error(error);
  process.exitCode = 1;
});

child.once("exit", (code, signal) => {
  clearTimeout(timer);
  if (forcedExitTimer) clearTimeout(forcedExitTimer);
  if (timedOut) process.exitCode = 124;
  else if (signal) {
    console.error(`WebGPU bring-up child exited from signal ${signal}`);
    process.exitCode = 1;
  } else process.exitCode = code ?? 1;
});
