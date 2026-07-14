/**
 * Maps a wall-clock-paced canvas recording back onto the simulation clock.
 *
 * A solver that needs 12 seconds to calculate 3 seconds of physics should be
 * replayed at 4x. The resulting 3-second presentation then preserves SI time:
 * one second in the video represents one simulated second.
 */
export function realTimePlaybackRate(recordedDuration_s: number, simulationDuration_s: number): number {
  if (!Number.isFinite(recordedDuration_s) || !Number.isFinite(simulationDuration_s)) return 1;
  if (recordedDuration_s <= 0 || simulationDuration_s <= 0) return 1;
  return recordedDuration_s / simulationDuration_s;
}
