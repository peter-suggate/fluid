/** Takes the GPU for a probe's steps after its solver is built, so shader
 * compilation (CPU) runs outside the shared lock. A .sh file is the shared
 * GPU runner (`<runner> <mode> <command>`, check unless a timing run is
 * asked for); any other file is taken with lockf. The hold is a child that keeps the lock until the returned release
 * or this process's exit. */
import { resolve } from "node:path";
import { spawn } from "node:child_process";
import { existsSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";

export async function takeGpuHold(file: string, mode: "check" | "timing" = "check"): Promise<() => void> {
  const flag = resolve(tmpdir(), `gpu-hold-${process.pid}`);
  const hold = ["sh", "-c", `touch '${flag}'; while [ -e '${flag}' ] && kill -0 ${process.pid} 2>/dev/null; do sleep 0.2; done; rm -f '${flag}'`];
  if (file.endsWith(".sh")) spawn(file, [mode, ...hold], { stdio: "ignore" }); else spawn("lockf", [file, ...hold], { stdio: "ignore" });
  while (!existsSync(flag)) await new Promise(r => setTimeout(r, 200));
  return () => { try { unlinkSync(flag); } catch { /* released */ } };
}
