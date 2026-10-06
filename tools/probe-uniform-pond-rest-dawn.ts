// Resting-pond probe (see runPondRestArm in ./uniform-pond-rest.ts for the options).
//   node --import tsx tools/probe-uniform-pond-rest-dawn.ts --arm=name [--frames=30] [options]
//   --batch='[["--arm=a","--frames=3"],["--arm=b"]]' runs several arms on one
//   device (one cold build); each entry's options override the shared ones.
import { runPondRestArm, withPondRestDevice } from "./uniform-pond-rest";

const shared = process.argv.slice(2).filter(a => !a.startsWith("--batch="));
const batch = process.argv.find(a => a.startsWith("--batch="))?.slice(8);
const arms: string[][] = batch ? JSON.parse(batch) : [[]];
const failures: string[] = [];
await withPondRestDevice(`uniform pond rest ${arms.length > 1 ? `${arms.length} arms` : shared.join(" ")}`, async device => {
  for (const entry of arms) {
    // A batch reports every arm: a fatal frame in one is that arm's result.
    try { await runPondRestArm(device, [...shared, ...entry]); }
    catch (error) {
      if (!batch) throw error;
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`${entry.join(" ")}: ${message}`); console.log(JSON.stringify({ arm: entry.find(a => a.startsWith("--arm="))?.slice(6), failed: message.slice(0, 600) }));
    }
  }
});
if (failures.length) { console.error(failures.join("\n")); process.exit(1); }
