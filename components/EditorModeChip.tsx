"use client";

import { DEFAULT_EDITOR_TOOL, getEditorTool } from "../lib/core/editor-tools";
import { simulation } from "../lib/core/simulation/controller";
import { useEditorHistoryStore } from "../lib/core/stores/history-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";

/**
 * What the next click will do, and how to stop doing it.
 *
 * This replaces the left-edge tool strip. A strip of mode buttons costs the
 * viewport a permanent column in order to answer a question that only has an
 * interesting answer while a mode is armed — the rest of the time it lists nine
 * things you are not doing. The ring answers "what can I do to *this*" at the
 * pointer, and the only state left to report is the one mode that is currently
 * on, so that is all this draws.
 *
 * Carrying outranks the armed tool because it is the more surprising state: the
 * pointer is holding an object, every camera gesture is suspended, and the
 * click that would normally select is now the release. That has to be legible
 * without the reader having to remember how they got here.
 *
 * UNDO and REDO stay because they are not modes — they are the two commands a
 * direct-manipulation editor cannot make discoverable any other way, and the
 * ring is contextual on a target so a history that spans targets does not
 * belong in it.
 */
export function EditorModeChip() {
  const activeTool = useUIStore((state) => state.activeTool);
  const setActiveTool = useUIStore((state) => state.setActiveTool);
  const carry = useUIStore((state) => state.carry);
  // A carry started by a selection carries no name — the UI store must not read
  // the document to get one — so the chip resolves it here, where the document
  // is already a legitimate dependency.
  const carriedName = useSceneStore((state) =>
    state.scene.rigidBodies.find((body) => body.id === carry?.bodyId)?.name);
  const canUndo = useEditorHistoryStore((state) => state.past.length > 0);
  const canRedo = useEditorHistoryStore((state) => state.future.length > 0);
  const tool = getEditorTool(activeTool);
  const armed = activeTool !== DEFAULT_EDITOR_TOOL;

  return (
    <div className="editor-mode-chip" data-active-tool={activeTool} data-carrying={Boolean(carry)}>
      {carry ? (
        <div className="mode-chip carrying" role="status" aria-live="polite" data-testid="carry-chip">
          <strong>{carry.label ?? carriedName ?? "carrying"}</strong>
          {carry.tiltDegrees !== 0 && <em>{carry.tiltDegrees > 0 ? "+" : ""}{Math.round(carry.tiltDegrees)}°</em>}
          <span>drag to dip and lift · wheel for depth · Q/E to pour · shift to ease · click to put down</span>
        </div>
      ) : armed ? (
        <button
          type="button"
          className="mode-chip armed"
          data-testid={`editor-mode-${activeTool}`}
          title={tool.hint}
          onClick={() => setActiveTool(DEFAULT_EDITOR_TOOL)}
        >
          <strong>{tool.label}</strong>
          <span>{tool.hint}</span>
          <em>esc</em>
        </button>
      ) : (
        <p className="mode-chip resting">right-click anything for what it can do</p>
      )}
      <div className="editor-history" role="group" aria-label="Edit history">
        <button type="button" disabled={!canUndo} title="Undo (⌘Z)" onClick={() => simulation.undo()}>UNDO</button>
        <button type="button" disabled={!canRedo} title="Redo (⇧⌘Z)" onClick={() => simulation.redo()}>REDO</button>
      </div>
    </div>
  );
}
