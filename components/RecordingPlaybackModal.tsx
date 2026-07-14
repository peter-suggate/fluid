"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { realTimePlaybackRate, sourceDurationForPlayback } from "@/lib/recording-timing";
import { simulationRecording } from "@/lib/simulation/recording";
import { useRecordingStore } from "@/lib/stores/recording-store";

type PlaybackMode = "real-time" | "source";

function applyPlaybackRate(video: HTMLVideoElement, rate: number) {
  try {
    video.defaultPlaybackRate = rate;
    video.playbackRate = rate;
  } catch {
    // The clock synchronizer below still enforces the requested timing when a
    // browser rejects unusually high native playback-rate values.
    video.defaultPlaybackRate = 1;
    video.playbackRate = 1;
  }
}

export function RecordingPlaybackModal() {
  const open = useRecordingStore((state) => state.modalOpen);
  const recording = useRecordingStore((state) => state.recording);
  const videoRef = useRef<HTMLVideoElement>(null);
  const playbackAnchorRef = useRef<{ wall_ms: number; media_s: number } | null>(null);
  const [mode, setMode] = useState<PlaybackMode>("real-time");
  const [mediaDuration_s, setMediaDuration_s] = useState(0);
  const simulationPaced = recording?.timingMode === "simulation-frames";
  const sourceDuration_s = useMemo(() => recording
    ? sourceDurationForPlayback(mediaDuration_s, recording.recordedDuration_s)
    : 0, [mediaDuration_s, recording]);
  const playbackRate = useMemo(() => recording
    ? simulationPaced ? 1 : realTimePlaybackRate(sourceDuration_s, recording.simulationDuration_s)
    : 1, [sourceDuration_s, recording]);

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
    applyPlaybackRate(video, rate);
    playbackAnchorRef.current = video.paused ? null : { wall_ms: performance.now(), media_s: video.currentTime };
  }, [mode, playbackRate]);

  // Some browsers accept a high playbackRate but silently decode at a lower
  // rate. Keep the media clock aligned to the requested simulation clock so a
  // slow solve still completes playback in exactly simulationDuration_s.
  useEffect(() => {
    if (!open || mode !== "real-time" || !recording || simulationPaced) return;
    let frame = 0;
    const keepRealTime = (now: number) => {
      const video = videoRef.current;
      if (video && !video.paused && !video.ended) {
        const anchor = playbackAnchorRef.current ?? { wall_ms: now, media_s: video.currentTime };
        playbackAnchorRef.current = anchor;
        const expected_s = Math.min(sourceDuration_s, anchor.media_s + (now - anchor.wall_ms) / 1000 * playbackRate);
        if (Math.abs(video.currentTime - expected_s) > 0.25) video.currentTime = expected_s;
      }
      frame = requestAnimationFrame(keepRealTime);
    };
    frame = requestAnimationFrame(keepRealTime);
    return () => cancelAnimationFrame(frame);
  }, [open, mode, playbackRate, recording, simulationPaced, sourceDuration_s]);

  const anchorPlayback = (video: HTMLVideoElement) => {
    const rate = mode === "real-time" ? playbackRate : 1;
    applyPlaybackRate(video, rate);
    playbackAnchorRef.current = { wall_ms: performance.now(), media_s: video.currentTime };
  };

  if (!open || !recording) return null;

  return (
    <div className="recording-modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) simulationRecording.close();
    }}>
      <section className="recording-modal" role="dialog" aria-modal="true" aria-labelledby="recording-title" data-testid="recording-playback-modal">
        <header>
          <div>
            <p className="eyebrow">SIMULATION-TIME CAPTURE</p>
            <h2 id="recording-title">Real-time playback{simulationPaced ? " · 30 fps" : ""}</h2>
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
            onLoadedMetadata={(event) => {
              const duration = event.currentTarget.duration;
              setMediaDuration_s(Number.isFinite(duration) && duration > 0 ? duration : 0);
              anchorPlayback(event.currentTarget);
            }}
            onDurationChange={(event) => {
              const duration = event.currentTarget.duration;
              if (Number.isFinite(duration) && duration > 0) setMediaDuration_s(duration);
            }}
            onPlay={(event) => anchorPlayback(event.currentTarget)}
            onPause={() => { playbackAnchorRef.current = null; }}
            onSeeked={(event) => anchorPlayback(event.currentTarget)}
          />
          <span className="recording-time-badge">{simulationPaced ? "SIMULATION-SAMPLED · 30 FPS · NATIVE ×1" : "1 VIDEO SECOND = 1 SIMULATION SECOND"}</span>
        </div>
        <div className="recording-playback-controls">
          <div className="segmented" aria-label="Playback timing">
            {simulationPaced
              ? <button className="active">Real time · 30 fps · ×1</button>
              : <><button className={mode === "real-time" ? "active" : ""} onClick={() => setMode("real-time")}>Real time · ×{playbackRate.toFixed(2)}</button><button className={mode === "source" ? "active" : ""} onClick={() => setMode("source")}>Original capture · ×1</button></>}
          </div>
          <button className="quiet-button" onClick={() => simulationRecording.download()} title={simulationPaced ? "Download the simulation-paced MP4" : "Download the original wall-clock-paced WebM"}>Download {simulationPaced ? "MP4" : "source"}</button>
        </div>
        <dl className="recording-stats">
          <div><dt>Simulation interval</dt><dd>{recording.simulationStart_s.toFixed(2)}–{recording.simulationEnd_s.toFixed(2)} s</dd></div>
          <div><dt>Real-time result</dt><dd>{recording.simulationDuration_s.toFixed(2)} s</dd></div>
          <div><dt>{simulationPaced ? "Captured frames" : "Source capture"}</dt><dd>{simulationPaced ? recording.frameCount?.toLocaleString() : `${sourceDuration_s.toFixed(2)} s`}</dd></div>
          <div><dt>{simulationPaced ? "Frame pacing" : "Timing correction"}</dt><dd>{simulationPaced ? `${recording.frameRate} fps · ×1` : `×${playbackRate.toFixed(2)}`}</dd></div>
        </dl>
        <p className="recording-note">{simulationPaced
          ? "Each frame was sampled at a 0.033 s simulation-time boundary and encoded consecutively at 30 fps. Playback and the downloaded MP4 therefore run natively at real-world speed without frame skipping or timeline seeking."
          : "This browser used compatibility capture. In-app playback is calibrated from simulated seconds; the downloaded WebM preserves its original wall-clock timing."}</p>
      </section>
    </div>
  );
}
