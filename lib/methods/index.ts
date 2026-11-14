import { installSimulationMethods } from "../core/method-registry";
import { seepMethod } from "./seep-method";
import { uniformNarrowBandMethod } from "./uniform/uniform-narrow-band-method";

/** The composition root installs the solver; consumers use the plugin registry.
 * Seep · M1 Max is experimental and opens only by URL (?method=seep-m1max). */
installSimulationMethods({
  methods: [uniformNarrowBandMethod, seepMethod],
  defaultId: uniformNarrowBandMethod.id,
});
