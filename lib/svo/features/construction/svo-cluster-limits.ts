/**
 * Dependency-leaf limits for seeded-lobe cluster authoring.
 *
 * Scene generators import these values while `svo-primitive-abi` imports the
 * generators indirectly through its rendering dependencies. Keeping the limits
 * in this leaf avoids making module initialization order part of the ABI.
 */

/** Fewer lobes degenerates to the centred ellipsoid; twelve bounds shader cost. */
export const SVO_CLUSTER_LOBE_MINIMUM_COUNT = 4;
export const SVO_CLUSTER_LOBE_MAXIMUM_COUNT = 12;

/** Largest supported ratio between a lobe's longest and shortest half-axis. */
export const SVO_CLUSTER_LOBE_MAXIMUM_ANISOTROPY = 4;
