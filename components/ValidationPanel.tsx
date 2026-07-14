"use client";

import type { ValidationResult } from "@/lib/validation";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";

export function ValidationPanel({ results }: { results: ValidationResult[] }) {
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const setValidationOpen = useUIStore((state) => state.setValidationOpen);
  const passed = results.filter((result) => result.passed).length;
  return (
    <section className="validation-panel" aria-label="Numerical validation report" data-testid="validation-panel">
      <header>
        <div><p className="eyebrow">STAGES 3–4 · NUMERICAL CONTRACT</p><h2>{passed}/{results.length} in-app checks passed</h2></div>
        <button className="icon-button" onClick={() => setValidationOpen(false)} aria-label="Close validation report">×</button>
      </header>
      <div className="validation-summary">
        <span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} />
        <div><strong>GPU capability</strong><small>{gpuStatus.label}</small></div>
      </div>
      <div className="validation-list">
        {results.map((result) => (
          <article key={result.id} className={result.passed ? "pass" : "fail"}>
            <span className="result-mark">{result.passed ? "PASS" : "FAIL"}</span>
            <div><strong>{result.id} · {result.name}</strong><small>Measured {result.measured} · Acceptance {result.threshold}</small></div>
          </article>
        ))}
      </div>
      <p className="validation-note">The regression suite gates rigid bodies, the Eulerian MAC oracle, buoyancy, sinking, and conservative two-way impulse exchange. WebGPU supplies the high-resolution interactive Eulerian path.</p>
    </section>
  );
}
