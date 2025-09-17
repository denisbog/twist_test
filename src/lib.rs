use cv::nalgebra::Point3;
/// A small helper: given camera intrinsics (fx, fy, cx, cy) and a pixel (u,v),
/// return a normalized bearing Unit<Vector3<f64>> pointing along (x,y,1) in camera coordinates.
pub fn pixel_to_bearing(u: f64, v: f64, fx: f64, fy: f64, cx: f64, cy: f64) -> Point3<f64> {
    let x = (u - cx) / fx;
    let y = (v - cy) / fy;
    Point3::new(x, y, 1.0)
}

/// Project a world point into pixel coordinates given a WorldToCamera pose and intrinsics.
/// Here we accept the WorldToCamera returned by lambda-twist (type: cv_core::WorldToCamera).
/// We compute camera_point = pose * world_point, then pixel = (fx*X/Z + cx, fy*Y/Z + cy).
pub fn project_world_point_to_pixel(
    pose: &cv::WorldToCamera,
    world_pt: &Point3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Option<(f64, f64)> {
    // WorldToCamera maps WorldPoint -> CameraPoint (camera coords, where Z is depth forward).
    // The cv_core WorldToCamera is a wrapper over an Isometry; it implements Projective/Pose stuff.
    // Here we'll transform the Point3 using the pose (pose.0 is the Isometry inside).
    let iso = &pose.0; // IsometryMatrix3<f64>
    let camera_point = iso.transform_point(world_pt);
    let x = camera_point.x;
    let y = camera_point.y;
    let z = camera_point.z;

    if z <= 0.0 {
        // point behind camera or at zero depth: skip
        return None;
    }

    let u = fx * (x / z) + cx;
    let v = fy * (y / z) + cy;
    Some((u, v))
}
