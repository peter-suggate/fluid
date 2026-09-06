"use client";

import { useState } from "react";
import type { SimulationFailure } from "../lib/core/simulation-failure";

export function SimulationFailureDetails({ failure, configuration, showSummary = true }: {
  failure: SimulationFailure;
  configuration: unknown;
  showSummary?: boolean;
}) {
  const [copyStatus, setCopyStatus] = useState("");
  const report = JSON.stringify({ version: 1, failure, configuration }, null, 2);
  const download = () => {
    const url = URL.createObjectURL(new Blob([report], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `simulation-failure-${failure.code}-frame-${failure.frame}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div data-testid="simulation-failure-details">
    {showSummary && <><strong>SIMULATION HALTED · {failure.code}</strong>
      <p>{failure.message}</p></>}
    <p>Frame {failure.frame} · generation {failure.generation} · owner {failure.ownerId}</p>
    <code>{failure.kernel}</code>
    <p>The first failure is retained. Further simulation work is blocked. Save this report before restarting.</p>
    <button type="button" onClick={async () => {
      try { await navigator.clipboard.writeText(report); setCopyStatus("Report copied"); }
      catch { setCopyStatus("Copy unavailable. Download the report or select the details below."); }
    }}>Copy failure report</button>
    <button type="button" onClick={download}>Download failure report</button>
    <span role="status">{copyStatus}</span>
    <details><summary>Debugging details and scene inputs</summary>
      <pre style={{ maxHeight: "40vh", overflow: "auto", whiteSpace: "pre-wrap" }}>{report}</pre>
    </details>
  </div>;
}
