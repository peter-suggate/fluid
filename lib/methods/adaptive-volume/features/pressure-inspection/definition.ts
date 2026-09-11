import type { MethodParamSpec, MethodParamValues } from "../../../../core/method-contract";
import { pressureJournalSchedule, type PressureJournalDescriptor } from "../../../../features/pressure-inspection/journal";
export const pressureCaptureParam: MethodParamSpec = {
    kind: "select",
    key: "pressureJournal",
    label: "Pressure film capture",
    default: "off",
    tier: "fine",
    options: [
      { value: "off", label: "Off" },
      { value: "on", label: "On · reserve the film" },
    ],
    // Structural, deliberately: the reservation is a region of the state buffer
    // and it is not small, so it cannot appear and disappear under a live
    // solver. Turning it on rebuilds once, which is the honest price of a
    // capture that is otherwise free — an armed frame is the only frame that
    // encodes a snapshot dispatch, and an unarmed one costs literally nothing.
    hint: "Reserves room to film one pressure solve, so the Pressure lab views can replay its iterations. About 192 bytes per pressure cell — a few megabytes on the mini scenes and hundreds on a large one — which is why it is off unless asked for. Reserving is not capturing: the snapshots are only written on frames a Pressure lab view is open, and the reservation alone changes no dispatch.",
  };
export function pressureCaptureDescriptor(iterations: (value: unknown) => number, snapshots: number): PressureJournalDescriptor<MethodParamValues> {
  return {
    isReserved: values => values.pressureJournal === "on",
    schedule: values => values.pressureJournal === "on" ? pressureJournalSchedule(iterations(values.pressureIterations), snapshots) : [],
    reserve: { parameter: "pressureJournal", value: "on" },
  };
}

/**
 * Capacity of the optional pressure journal, chosen at construction.
 *
 * Sized here rather than armed at runtime because the journal is a tail range
 * of the resident state buffer, which exists once. Defaulting it on would be
 * the "capacity is not inert" mistake: on a large scene the snapshot region is
 * tens of megabytes, and a lane that never opens the pressure lab would pay it
 * at t=0. So the default is zero floats and zero dispatches, and a caller that
 * wants the film asks for it.
 */
export interface SparseCM12PressureJournalCapacityRequest {
  /**
   * Encoded iterations to reserve records for. Pass the solve's iteration
   * ceiling; the reservation adds one for the seed record.
   */
  readonly iterationCapacity?: number;
  /** Whole-field snapshots to reserve. Twelve covers a 128-iteration solve. */
  readonly snapshotCapacity?: number;
}

/** Snapshots reserved when a caller asks for a journal without saying how many. */
export const SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS = 12;
