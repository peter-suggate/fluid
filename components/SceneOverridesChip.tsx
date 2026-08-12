"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { simulation } from "@/lib/simulation/controller";
import {
  cleanSceneLink,
  countedSceneOverrides,
  sceneHasPresetBaseline,
  sceneOverridesInQuery,
  type SceneOverride,
  type SceneOverrideGroup,
} from "@/lib/scene-overrides";
import { useMethodStore } from "@/lib/stores/method-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useShellStore } from "@/lib/stores/shell-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { createSceneQueryLayerCache, replaceQueryStateUrl, serializeQueryState } from "@/lib/url-state";

/**
 * One cache for the process, matching `startQueryStateSync`'s own: the scene
 * layer of the query is derived from an immutable document, so re-deriving it
 * for a camera orbit would re-serialize a sculpted terrain at pointer rate.
 */
const cachedSceneLayer = createSceneQueryLayerCache();

/** Hydration, as an external store: the client snapshot is true, the server's false. */
const subscribeNever = () => () => {};

const GROUP_LABELS: Readonly<Record<SceneOverrideGroup, string>> = {
  scene: "Scene document",
  method: "Solver",
  render: "Presentation",
  view: "Also in the link",
  link: "Startup flags",
};

/**
 * Keys a reload has to retire, removed from the address on the way out.
 *
 * The canonical write first, and not incidentally: the stores mirror themselves
 * into the address a microtask later, so the live location is still the one
 * from *before* the clear. Navigating to that minus one key would reload every
 * override this clear had just removed — the flag would go and nothing else
 * would, which is the exact opposite of what the button says.
 */
function reloadWithout(keys: readonly string[]) {
  replaceQueryStateUrl();
  const query = new URLSearchParams(window.location.search);
  for (const key of keys) query.delete(key);
  const search = query.toString();
  window.location.replace(`${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`);
}

function OverrideRow({ override, onClear }: { override: SceneOverride; onClear: () => void }) {
  return (
    <li data-group={override.group} data-counted={override.counted}>
      <span className="override-label" title={override.hint}>{override.label}</span>
      <span className="override-value" title={override.value}>{override.value}</span>
      <button
        type="button"
        className="reset-chip"
        onClick={onClear}
        title={override.clearedBy === "reload"
          ? `Remove ${override.keys.join(", ")} and reload`
          : "Restore the authored value"}
        aria-label={`Clear ${override.label}`}
      >↺</button>
    </li>
  );
}

/**
 * What this scene is carrying that its author did not put there, and the way
 * out of it.
 *
 * A studio whose entire state is a link is a studio a reader can arrive in
 * without knowing what was done to it: a solver parameter someone was A/B'ing,
 * a debug stage view, a `gpu=off` from a bug report. Every one of those already
 * had a control somewhere in the panels, and none of them said *that it was
 * not the default* from the one place always on screen. So the count sits
 * beside the scene name, and the list beneath it is the same diff the address
 * bar is — see `sceneOverridesInQuery`.
 *
 * The camera and the open panel are listed but never counted. They are in the
 * link so a shared link reopens the view that was shared; treating them as
 * overrides would leave the badge lit from the first orbit and it would stop
 * meaning anything.
 */
export function SceneOverridesChip() {
  const scene = useSceneStore((state) => state.scene);
  const presetId = useSceneStore((state) => state.presetId);
  const methodState = useMethodStore();
  const uiState = useUIStore();
  const shellView = useShellStore((state) => state.view);
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  // The query is only canonical in the browser, and the studio's first paint is
  // hydrated: read the address after mount so the server and the client agree
  // on an empty chip and the address is consulted only once they do.
  const mounted = useSyncExternalStore(subscribeNever, () => true, () => false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  if (!mounted) return null;
  const search = serializeQueryState(
    window.location.search,
    { presetId, scene },
    methodState,
    uiState,
    { view: shellView },
    cachedSceneLayer({ presetId, scene }),
  );
  const overrides = sceneOverridesInQuery(search, { hasPresetBaseline: sceneHasPresetBaseline(presetId) });
  const counted = countedSceneOverrides(overrides);
  if (overrides.length === 0) return null;

  const clear = (keys: readonly string[]) => {
    const { reload } = simulation.clearOverrides(keys);
    if (reload.length > 0) reloadWithout(reload);
  };
  const groups = (["scene", "method", "render", "link", "view"] as const)
    .map((group) => [group, overrides.filter((override) => override.group === group)] as const)
    .filter(([, rows]) => rows.length > 0);

  return (
    <div className="scene-overrides" ref={rootRef} data-testid="scene-overrides">
      <button
        type="button"
        className={`scene-overrides-chip${counted.length > 0 ? " modified" : ""}${open ? " active" : ""}`}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        data-testid="scene-overrides-chip"
        data-override-count={counted.length}
        title={counted.length > 0
          ? `${counted.length} setting${counted.length === 1 ? "" : "s"} in this link differ${counted.length === 1 ? "s" : ""} from the authored scene`
          : "This link carries a camera or panel state, but no overridden settings"}
      >
        <i aria-hidden="true" />
        <strong>{counted.length > 0 ? `${counted.length} OVERRIDE${counted.length === 1 ? "" : "S"}` : "LINK STATE"}</strong>
      </button>
      {open && <div className="scene-overrides-popover" role="dialog" aria-label="Link overrides">
        <header>
          <strong>{counted.length > 0 ? "This scene is not as authored" : "Nothing is overridden"}</strong>
          <span>Everything the address bar carries beyond this scene&apos;s identity.</span>
        </header>
        {groups.map(([group, rows]) => (
          <section key={group} data-group={group}>
            <h4>{GROUP_LABELS[group]}</h4>
            <ul>
              {rows.map((override) => (
                <OverrideRow key={override.keys.join("|")} override={override} onClear={() => clear(override.keys)} />
              ))}
            </ul>
          </section>
        ))}
        <footer>
          <button
            type="button"
            className="quiet-button"
            onClick={() => {
              void navigator.clipboard?.writeText(new URL(cleanSceneLink(presetId), window.location.origin).href);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1400);
            }}
            title="A link to this scene exactly as it was authored"
          >{copied ? "Copied" : "Copy clean link"}</button>
          <button
            type="button"
            className="primary-button"
            data-testid="clear-scene-overrides"
            onClick={() => { clear(overrides.flatMap((override) => override.keys)); setOpen(false); }}
          >Clear all</button>
        </footer>
        <small>Restoring is one undo step. Authoring a URL cannot hold — a sculpted terrain, added
          scenery — is untouched; reopen the scene from the library to discard that too.</small>
      </div>}
    </div>
  );
}
