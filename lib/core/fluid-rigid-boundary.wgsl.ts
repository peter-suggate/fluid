/** Signed shape codes distinguish obstacles from closed fluid domains.
 * -(shape + 1) reverses the solid side without changing the shape table. */
export const fluidRigidBoundaryWGSL = /* wgsl */ `
fn fluidRigidInside(code:i32,dimensions:vec3f,local:vec3f)->bool{
 let inside=rigidShapeInside(select(code,-code-1,code<0),dimensions,local);
 return select(inside,!inside,code<0);
}
fn fluidRigidDistance(code:i32,dimensions:vec3f,local:vec3f)->f32{
 return select(1.0,-1.0,code<0)*rigidShapeDistance(select(code,-code-1,code<0),dimensions,local);
}
`;
