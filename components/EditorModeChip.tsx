"use client";

import { getEditorGesture } from "../lib/core/editor-gesture-catalog";
import { useSession } from "../lib/core/session/session-context";

/**
 * What the next drag will do, and how to stop doing it.
 *
 * This replaces the left-edge tool strip. A strip of mode buttons costs the
 * viewport a permanent column in order to answer a question that only has an
 * interesting answer while a mode is armed — the rest of the time it lists nine
 * things you are not doing. The ring answers "what can I do to *this*" at the
 * pointer, and the only state left to report is the one mode that is currently
 * on, so that is all this draws.
 *
 * Carrying outranks the armed gesture because it is the more surprising state: the
 * pointer is holding an object, every camera gesture is suspended, and the
 * click that would normally select is now the release. That has to be legible
 * without the reader having to remember how they got here.
 *
 * Nothing at rest. The resting hint told a reader to right-click once and then
 * sat on the image forever, and the UNDO/REDO pair was two permanent buttons for
 * a command every editor already binds to a key. Both are gone: this is a state
 * report, and with no mode armed and nothing in hand there is no state.
 */
export function EditorModeChip() {
  const session = useSession();
  const armedGesture = session.ui((state) => state.armedGesture);
  const setArmedGesture = session.ui((state) => state.setArmedGesture);
  const carry = session.ui((state) => state.carry);
  // A carry started by a selection carries no name — the UI store must not read
  // the document to get one — so the chip resolves it here, where the document
  // is already a legitimate dependency.
  const carriedName = session.scene((state) =>
    state.scene.rigidBodies.find((body) => body.id === carry?.bodyId)?.name);
  const gesture = armedGesture ? getEditorGesture(armedGesture) : undefined;
  if (!carry && !gesture) return null;

  return (
    <div className="editor-mode-chip" data-armed-gesture={armedGesture} data-carrying={Boolean(carry)}>
      {carry ? (
        <div className="mode-chip carrying" role="status" aria-live="polite" data-testid="carry-chip">
          <strong>{carry.label ?? carriedName ?? "carrying"}</strong>
          {carry.tiltDegrees !== 0 && <em>{carry.tiltDegrees > 0 ? "+" : ""}{Math.round(carry.tiltDegrees)}°</em>}
          <span>drag to dip and lift · wheel for depth · Q/E to pour · shift to ease · click to put down</span>
        </div>
      ) : (
        <button
          type="button"
          className="mode-chip armed"
          data-testid={`editor-mode-${armedGesture}`}
          title={gesture!.hint}
          onClick={() => setArmedGesture(undefined)}
        >
          <strong>{gesture!.label}</strong>
          <span>{gesture!.hint}</span>
          <em>esc</em>
        </button>
      )}
    </div>
  );
}
