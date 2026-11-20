import { installSimulationMethods } from "../core/method-registry";
import { pourMethod } from "./pour-method";
import { seepMethod } from "./seep-method";
import { uniformNarrowBandMethod } from "./uniform/uniform-narrow-band-method";

/** The composition root installs the solver; consumers use the plugin registry.
 * Seep · M1 Max and Pour · M1 Max are experimental single-scene solvers: the toolstrip's "Fluid solver" picker lists
 * them beside NB-FLIP's engines, enabled on their own scene only (also ?method=seep-m1max, ?method=pour-m1max). */
installSimulationMethods({
  methods: [uniformNarrowBandMethod, seepMethod, pourMethod],
  defaultId: uniformNarrowBandMethod.id,
});
