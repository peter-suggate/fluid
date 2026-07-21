"use client";

import { useEffect, useState } from "react";
import { resetPresentationFrameRate, samplePresentationFrameRate } from "@/lib/presentation-frame-rate";

const SAMPLE_INTERVAL_MS = 500;

export function FrameRateCounter() {
  const [fps, setFps] = useState<number | null>(null);

  useEffect(() => {
    resetPresentationFrameRate();
    const timer = window.setInterval(() => setFps(samplePresentationFrameRate(performance.now())), SAMPLE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, []);

  const roundedFps = fps === null ? null : Math.round(fps);
  return <output
    className="frame-rate-counter"
    aria-label={roundedFps === null ? "Presentation frame rate unavailable" : `Presentation frame rate ${roundedFps} frames per second`}
    title="Actual WebGPU presentation submissions per second"
    data-testid="frame-rate-counter"
  >
    <strong>{roundedFps ?? "—"}</strong><span>FPS</span>
  </output>;
}
