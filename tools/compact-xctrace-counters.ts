/** Compact an already-exported xctrace counter table in place.
 *
 * The replacement is built beside the artifact and renamed only after a
 * complete parse, so interruption leaves the source table untouched. The raw
 * .trace remains the reproducible full-rate authority.
 */
import { createWriteStream, existsSync, renameSync, rmSync, statSync } from "node:fs";
import { once } from "node:events";
import { resolve } from "node:path";
import { writeFile } from "node:fs/promises";
import { readTraceRows } from "./xctrace-trace-tables";
import { CounterRowSelector, makeCounterExtractionPolicy } from "./profile-mini-dam-xctrace";
import "../lib/methods";

const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);

const main = async (): Promise<void> => {
  const artifact = resolve(flag("artifact") ?? "");
  const targetReduction = Number(flag("counter-reduction") ?? 100);
  if (!flag("artifact")) throw new Error("--artifact=DIR is required");
  if (!Number.isFinite(targetReduction) || targetReduction < 1) {
    throw new Error("--counter-reduction must be at least 1");
  }
  const infoPath = resolve(artifact, "counter-info.ndjson");
  const sourcePath = resolve(artifact, "counter-value.ndjson");
  const compactPath = resolve(artifact, "counter-value.compacting.ndjson");
  if (!existsSync(infoPath) || !existsSync(sourcePath)) {
    throw new Error(`counter-info.ndjson and counter-value.ndjson are required in ${artifact}`);
  }

  const countersById = new Map<string, string>();
  const sampleIntervals = new Set<number>();
  for await (const row of readTraceRows(infoPath)) {
    countersById.set(String(row["counter-id"]), String(row.name ?? ""));
    const interval = Number(String(row["sample-interval"] ?? "0").replace(/,/g, ""));
    if (interval > 0) sampleIntervals.add(interval);
  }
  const policy = makeCounterExtractionPolicy(countersById, targetReduction);
  const selector = new CounterRowSelector(policy);
  const sink = createWriteStream(compactPath, { encoding: "utf8" });
  let sourceRows = 0;
  let retainedRows = 0;
  try {
    for await (const row of readTraceRows(sourcePath)) {
      sourceRows += 1;
      if (!selector.keep(row)) continue;
      retainedRows += 1;
      if (!sink.write(`${JSON.stringify(row)}\n`)) await once(sink, "drain");
    }
    sink.end();
    await once(sink, "finish");
  } catch (error) {
    sink.destroy();
    rmSync(compactPath, { force: true });
    throw error;
  }

  if (retainedRows === 0 || sourceRows / retainedRows < targetReduction) {
    rmSync(compactPath, { force: true });
    throw new Error(`compaction retained ${retainedRows}/${sourceRows} rows, below the requested`
      + ` ${targetReduction}x reduction`);
  }
  const sourceBytes = statSync(sourcePath).size;
  const retainedBytes = statSync(compactPath).size;
  const sourceBackup = resolve(artifact, "counter-value.full-rate.replacing.ndjson");
  renameSync(sourcePath, sourceBackup);
  renameSync(compactPath, sourcePath);
  rmSync(sourceBackup, { force: true });

  const hardwareSampleIntervalNs = sampleIntervals.size === 1
    ? [...sampleIntervals][0] : undefined;
  await writeFile(resolve(artifact, "counter-extraction.json"), `${JSON.stringify({
    targetReduction,
    sourceCounterCount: policy.sourceCounterCount,
    retainedCounterCount: policy.retainedCounterCount,
    retainedCounters: [...countersById]
      .filter(([id]) => policy.retainedCounterIds.has(id)).map(([, name]) => name),
    timestampStride: policy.timestampStride,
    hardwareSampleIntervalNs,
    retainedSampleIntervalNs: hardwareSampleIntervalNs === undefined ? undefined
      : hardwareSampleIntervalNs * policy.timestampStride,
    sourceRows,
    retainedRows,
    rowReduction: sourceRows / retainedRows,
    sourceBytes,
    retainedBytes,
    byteReduction: sourceBytes / retainedBytes,
    note: "Metal intervals, encoder labels, command-buffer boundaries, and shader samples are not downsampled.",
  }, null, 2)}\n`);
  console.log(`counter table: ${sourceRows} -> ${retainedRows} rows`
    + ` (${(sourceRows / retainedRows).toFixed(1)}x)`);
  console.log(`counter bytes: ${sourceBytes} -> ${retainedBytes}`
    + ` (${(sourceBytes / retainedBytes).toFixed(1)}x)`);
};

await main().catch((error: unknown) => {
  console.error(`counter compaction aborted: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
