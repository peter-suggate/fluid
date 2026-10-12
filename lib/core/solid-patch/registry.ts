import { createSolidPatchKindRegistry } from "./kind";
import { blockSolidPatchKind } from "./kinds/block";
import { boxSolidPatchKind } from "./kinds/box";
import { cylinderSolidPatchKind } from "./kinds/cylinder";
import { ellipsoidSolidPatchKind } from "./kinds/ellipsoid";

/** Every patch kind this build can bake. A new kind is its file and a line here. */
export const solidPatchKinds = createSolidPatchKindRegistry([
  boxSolidPatchKind,
  ellipsoidSolidPatchKind,
  cylinderSolidPatchKind,
  blockSolidPatchKind,
]);
