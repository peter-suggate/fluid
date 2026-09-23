/**
 * The app's control primitives and the rows that lay them out.
 *
 * Every panel — the studio flyouts, the pipeline cards, the toolstrip and its
 * panes, the labs — builds its controls from these, so a slider commits,
 * prints, disables and names itself the same way everywhere. See `controls.tsx`
 * for how a surface sizes them.
 */
export {
  Button, Choice, Metric, NumberInput, ResetButton, Select, Slider, Stepper, Switch, ToggleButton, Value,
  type ControlOption,
} from "./controls";
export {
  ChoiceField, ControlRow, Facts, Field, FieldActions, FieldList, FieldNote, NumberField, RangeField, Readout,
  SelectField, SwitchField,
} from "./fields";
export { clampNumber, formatNumber, printNumber, stepDecimals } from "./number";
export { useNumberEntry, useSliderGesture } from "./gestures";
