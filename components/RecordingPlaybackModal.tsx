"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { realTimePlaybackRate } from "@/lib/recording-timing";
import { simulationRecording } from "@/lib/simulation/recording";
import { useRecordingStore } from "@/lib/stores/recording-store";

type PlaybackMode = "real-time" | "source";

export function RecordingPlaybackModal() {
  const open = useRecordingStore((state) => state.modalOpen);
  const recording = useRecordingStore((state) => state.recording);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mode, setMode] = useState<PlaybackMode>("real-time");
  const [mediaDuration_s, setMediaDuration_s] = useState(0);
  const playbackRate = useMemo(() => recording
    ? realTimePlaybackRate(mediaDuration_s || recording.recordedDuration_s, recording.simulationDuration_s)
    : 1, [mediaDuration_s, recording]);

  useEffect(() => {
    if (!open) return;
    setMode("real-time");
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") simulationRecording.close();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const rate = mode === "real-time" ? playbackRate : 1;
    video.defaultPlaybackRate = rate;
    video.playbackRate = rate;
  }, [mode, playbackRate]);

  if (!open || !recording) return null;

  return (
    <div className="recording-modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) simulationRecording.close();
    }}>
      <section className="recording-modal" role="dialog" aria-modal="true" aria-labelledby="recording-title" data-testid="recording-playback-modal">
        <header>
          <div>
            <p className="eyebrow">SIMULATION-TIME CAPTURE</p>
            <h2 id="recording-title">Real-time playback</h2>
          </div>
          <button className="icon-button" onClick={() => simulationRecording.close()} aria-label="Close recording playback" autoFocus>×</button>
        </header>
        <div className="recording-video-frame">
          <video
            ref={videoRef}
            src={recording.url}
            controls
            autoPlay
            playsInline
            onLoadedMetadata={(event) => setMediaDuration_s(event.currentTarget.duration)}
            onPlay={(event) => { event.currentTarget.playbackRate = mode === "real-time" ? playbackRate : 1; }}
          />
          <span className="recording-time-badge">1 VIDEO SECOND = 1 SIMULATION SECOND</span>
        </div>
        <div className="recording-playback-controls">
          <div className="segmented" aria-label="Playback timing">
            <button className={mode === "real-time" ? "active" : ""} onClick={() => setMode("real-time")}>Real time · ×{playbackRate.toFixed(2)}</button>
            <button className={mode === "source" ? "active" : ""} onClick={() => setMode("source")}>Original capture · ×1</button>
          </div>
          <button className="quiet-button" onClick={() => simulationRecording.download()} title="Download the original wall-clock-paced WebM">Download source</button>
        </div>
        <dl className="recording-stats">
          <div><dt>Simulation interval</dt><dd>{recording.simulationStart_s.toFixed(2)}–{recording.simulationEnd_s.toFixed(2)} s</dd></div>
          <div><dt>Real-time result</dt><dd>{recording.simulationDuration_s.toFixed(2)} s</dd></div>
          <div><dt>Source capture</dt><dd>{(mediaDuration_s || recording.recordedDuration_s).toFixed(2)} s</dd></div>
          <div><dt>Timing correction</dt><dd>×{playbackRate.toFixed(2)}</dd></div>
        </dl>
        <p className="recording-note">Playback is paced from simulated seconds, not render time. Pauses are excluded, so motion under −9.8 m/s² is shown on a real-world clock even when the solver runs slowly. The downloaded WebM preserves the original capture timing; the calibrated view is available here whenever the capture remains open in this browser.</p>
      </section>
    </div>
  );
}
