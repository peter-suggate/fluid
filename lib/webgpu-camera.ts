/**
 * Perspective scale shared by every WebGPU presentation path.
 *
 * This is tan(verticalFieldOfView / 2), not an angle in radians. Keeping the
 * semantic in the name prevents raster and analytic-ray cameras diverging.
 */
export const CAMERA_TAN_HALF_FOV = 0.72;
