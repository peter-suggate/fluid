"use client";

import { useEffect, useState } from "react";
import { simulation } from "../lib/core/simulation/controller";
import {
  browserSceneLibraryStorage,
  deleteSceneFromLibrary,
  readSceneLibrary,
  renameSceneInLibrary,
  savedSceneEntries,
  type SceneLibraryEntry,
} from "../lib/core/scene-library";
import { useSceneStore } from "../lib/core/stores/scene-store";

function savedLabel(savedAt_ms: number): string {
  const elapsed_s = Math.max(0, (Date.now() - savedAt_ms) / 1000);
  if (elapsed_s < 60) return "just now";
  if (elapsed_s < 3600) return `${Math.floor(elapsed_s / 60)}m ago`;
  if (elapsed_s < 86_400) return `${Math.floor(elapsed_s / 3600)}h ago`;
  return `${Math.floor(elapsed_s / 86_400)}d ago`;
}

/** Save/load/rename/delete for authored scenes, as a configuration section. */
export function SceneLibraryPanel() {
  const sceneId = useSceneStore((state) => state.scene.sceneId);
  const [entries, setEntries] = useState<SceneLibraryEntry[]>([]);
  const [name, setName] = useState("");
  const [renaming, setRenaming] = useState<string | undefined>();

  // Storage is only readable in the browser, so the first list arrives after mount.
  useEffect(() => setEntries(readSceneLibrary(browserSceneLibraryStorage())), []);

  // The autosaved working document shares this storage but is not a scene the
  // reader saved; it is offered as Continue in the library, not renamed here.
  const saved = savedSceneEntries(entries);

  const save = () => {
    setEntries(simulation.saveNamedScene(name || sceneId));
    setName("");
  };

  return (
    <section className="scene-library" data-testid="scene-library">
      <div className="popover-section-heading"><h3>My scenes</h3><span>{saved.length} SAVED</span></div>
      <div className="scene-library-save">
        <input
          value={name}
          placeholder={sceneId}
          aria-label="Scene name"
          maxLength={80}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") save(); }}
        />
        <button type="button" onClick={save}>SAVE</button>
      </div>
      <div className="scene-library-list">
        {saved.map((entry) => (
          <div key={entry.id} className="scene-library-entry">
            {renaming === entry.id ? (
              <input
                autoFocus
                defaultValue={entry.name}
                aria-label={`Rename ${entry.name}`}
                onBlur={(event) => { setEntries(renameSceneInLibrary(browserSceneLibraryStorage(), entry.id, event.target.value)); setRenaming(undefined); }}
                onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); if (event.key === "Escape") setRenaming(undefined); }}
              />
            ) : (
              <button type="button" className="scene-library-load" title={`Load ${entry.name}`} onClick={() => simulation.loadNamedScene(entry)}>
                <strong>{entry.name}</strong><small>{savedLabel(entry.savedAt_ms)}</small>
              </button>
            )}
            <button type="button" className="quiet-button" aria-label={`Rename ${entry.name}`} onClick={() => setRenaming(entry.id)}>✎</button>
            <button type="button" className="quiet-button" aria-label={`Delete ${entry.name}`} onClick={() => setEntries(deleteSceneFromLibrary(browserSceneLibraryStorage(), entry.id))}>×</button>
          </div>
        ))}
        {saved.length === 0 && <p className="panel-note">Authored scenes you save appear here. They persist in this browser and export as the same JSON as the file download.</p>}
      </div>
    </section>
  );
}
