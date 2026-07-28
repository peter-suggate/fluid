/** Rebuild the self-contained HTML view from an already reduced xctrace summary. */
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { renderFrameReportHtml, type FrameReport } from "./xctrace-frame-report";

const requested = process.argv[2];
if (!requested) {
  throw new Error("usage: node --import tsx tools/rerender-xctrace-report.ts PATH/summary.json");
}
const summaryPath = resolve(requested);
const report = JSON.parse(await readFile(summaryPath, "utf8")) as FrameReport;
const outputPath = resolve(dirname(summaryPath), "report.html");
await writeFile(outputPath, renderFrameReportHtml(report));
console.log(outputPath);
