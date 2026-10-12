import { createVoxelRegionVerbRegistry } from "../verb";
import { clearVerb } from "./clear";
import { copyVerb } from "./copy";
import { cutVerb } from "./cut";
import { fillVerb } from "./fill";
import { flipVerb } from "./flip";
import { growVerb } from "./grow";
import { hollowVerb } from "./hollow";
import { nudgeVerb } from "./nudge";
import { paintVerb } from "./paint";
import { pasteVerb } from "./paste";
import { rotateVerb } from "./rotate";
import { deselectVerb, selectAllVerb } from "./select";
import { shrinkVerb } from "./shrink";

export const voxelRegionVerbs = createVoxelRegionVerbRegistry([
  clearVerb, fillVerb, paintVerb, hollowVerb, growVerb, shrinkVerb,
  flipVerb, rotateVerb, nudgeVerb,
  copyVerb, cutVerb, pasteVerb,
  selectAllVerb, deselectVerb,
]);
