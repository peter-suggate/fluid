"use client";
import { useState } from "react";
import { ScenePickerPopover } from "../components/ScenePickerPopover";
import { sceneCatalogCards } from "../lib/core/scenes";
import styles from "./AdvanceLab.module.css";

/** The same scene cards, search, recents and keyboard navigation as the studio. */
export function LabSceneSelector({
  sceneId,
  dimensions,
  choose,
  sliceLabel = "centre-Z slice",
}: {
  sceneId: string;
  dimensions?: readonly [number, number];
  choose: (id: string) => void;
  sliceLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className={styles.anchor}>
      <button
        type="button"
        className={styles.sceneChip}
        data-scene-selector-toggle=""
        aria-label="Choose scene"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <b>
          {sceneCatalogCards.find((card) => card.id === sceneId)?.name ??
            "Choose scene"}
        </b>
        {dimensions && (
          <em>
            {dimensions[0]}×{dimensions[1]} {sliceLabel}
          </em>
        )}
        <svg viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.6 5 6.6 8 3.6" />
        </svg>
      </button>
      {open && (
        <ScenePickerPopover
          className={styles.scenePopover}
          cards={sceneCatalogCards}
          currentId={sceneId}
          label="Choose the production scene this lab slices"
          choose={(card) => {
            choose(card.id);
            setOpen(false);
          }}
          close={() => setOpen(false)}
        />
      )}
    </div>
  );
}
