use cv::Estimator;
use cv::FeatureWorldMatch;
use cv::WorldPoint; // FeatureWorldMatch<P> and WorldPoint
use cv::nalgebra::Isometry3;
use cv::nalgebra::Matrix3;
use cv::nalgebra::Perspective3;
use cv::nalgebra::Point2;
use cv::nalgebra::Rotation3;
use cv::nalgebra::Translation3;
use cv::nalgebra::Vector2;
use cv::nalgebra::{Point3, Unit, Vector3};
use lambda_twist::LambdaTwist;
/// A small helper: given camera intrinsics (fx, fy, cx, cy) and a pixel (u,v),
/// return a normalized bearing Unit<Vector3<f64>> pointing along (x,y,1) in camera coordinates.
fn pixel_to_bearing(u: f64, v: f64, fx: f64, fy: f64, cx: f64, cy: f64) -> Unit<Vector3<f64>> {
    let x = (u - cx) / fx;
    let y = (v - cy) / fy;
    Unit::new_normalize(Vector3::new(x, y, 1.0))
}

/// Project a world point into pixel coordinates given a WorldToCamera pose and intrinsics.
/// Here we accept the WorldToCamera returned by lambda-twist (type: cv_core::WorldToCamera).
/// We compute camera_point = pose * world_point, then pixel = (fx*X/Z + cx, fy*Y/Z + cy).
fn project_world_point_to_pixel(
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

fn main() {
    // --- Example synthetic data -------------------------------------------------
    // Choose simple camera intrinsics:
    let fx = 4000.0;
    let fy = 3000.0;
    let cx = 2000.0;
    let cy = 1500.0;

    // Define three non-collinear world points (in meters).
    let w1 = Point3::new(7.54, 0.0, 2.75);
    let w2 = Point3::new(7.54, 0.0, 0.0);
    let w3 = Point3::new(5.82, 0.0, 0.85);

    let rot = Rotation3::from_euler_angles(
        89.3215f64.to_radians(),
        -2.71232f64.to_radians(),
        -110.38f64.to_radians(),
    );
    let trans = Translation3::new(0.536798, 3.3126, 1.31053);

    let global_transform = Isometry3::from_parts(trans, rot.into());
    println!("view {:?}", global_transform);

    let matrix_world = (global_transform).to_homogeneous();

    let coords = Point3::new(7.54, 0.0, 0.0);
    println!("coords : {coords}");
    let point = matrix_world.transform_point(&coords);
    println!("world point: {point}");

    println!("matrix_world: {matrix_world}");
    let projection = Perspective3::new(1.0, 101.49546f64.to_radians(), 0.1, 1000.0);

    let matrix_view = (global_transform).to_homogeneous().try_inverse().unwrap();
    println!("matrix_view:  {matrix_view}");
    let mut projection = projection.into_inner();

    *projection.index_mut((0, 2)) = 0.000485966;
    *projection.index_mut((1, 2)) = 0.0063451147;
    println!("projection: {projection}");
    let view_projection = projection * matrix_view;
    println!("view_projection: {view_projection}");
    let point = projection
        * global_transform
            .inverse_transform_point(&coords)
            .to_homogeneous();
    println!("view point: {point}");
    let point = Point3::from_homogeneous(point).unwrap();
    println!("view point: {point}");
    // Project world points with the ground-truth pose to produce pixel observations:
    let project_gt = |p: &Point3<f64>| {
        let cp = global_transform.inverse_transform_point(p); //moving the camera
        (fx * (cp.x / cp.z) + cx, fy * (cp.y / cp.z) + cy)
    };
    let (u1, v1) = project_gt(&w1);
    let (u2, v2) = project_gt(&w2);
    let (u3, v3) = project_gt(&w3);

    println!("Original pixel observations:"); // should be given
    println!("p1 = ({:.3}, {:.3})", u1, v1);
    println!("p2 = ({:.3}, {:.3})", u2, v2);
    println!("p3 = ({:.3}, {:.3})", u3, v3);

    [&w1, &w2, &w3].iter().for_each(|item| {
        let cp = global_transform.inverse_transform_point(item);
        println!("projection before normalized {}", cp.coords.normalize()); // to get here
        println!("projection before {cp}");
        let cp = projection.transform_point(&cp);
        println!("projection {cp}");
        let p = (fx * (cp.x / cp.z) + cx, -fy * (cp.y / cp.z) + cy);
        println!("p {p:?}");
        let p = (fx * (cp.x) / 2.0 + cx, -fx * (cp.y) / 2.0 + cy);
        println!("p {p:?}");

        let transform = Matrix3::new_nonuniform_scaling(&Vector2::new(fx / 2.0, -fx / 2.0))
            .append_translation(&Vector2::new(cx, cy));
        println!("image point before {cp:?}");
        let point = Point2::new(cp.x, cp.y).to_homogeneous();
        println!("point {point:?}");
        println!("transform {transform}");
        let p2 = Point2::from_homogeneous(transform * point).unwrap();

        println!("p2 {p2:?}");

        let test = Point3::from(
            (projection.try_inverse().unwrap()
                * Point3::from(transform.try_inverse().unwrap() * p2.to_homogeneous())
                    .to_homogeneous())
            .normalize()
            .xyz(),
        );
        println!("test {test}");
    });

    let coords = vec![
        Point3::new(7.54, 0.0, 2.75),
        Point3::new(7.54, 0.0, 0.0),
        Point3::new(5.82, 0.0, 0.85),
    ];

    let pixels = vec![
        Point2::new(2154.737265078878, 1194.8945899620421),
        Point2::new(2126.5841860517053, 1776.635153145754),
        Point2::new(2331.1681437869565, 1632.3580325758512),
    ];

    let bearings: Vec<Point3<f64>> = pixels
        .iter()
        .map(|item| {
            let transform = Matrix3::new_nonuniform_scaling(&Vector2::new(fx / 2.0, -fx / 2.0))
                .append_translation(&Vector2::new(cx, cy));
            Point3::from(
                (projection.try_inverse().unwrap()
                    * Point3::from(transform.try_inverse().unwrap() * item.to_homogeneous())
                        .to_homogeneous())
                .normalize()
                .xyz(),
            )
        })
        .collect();

    // --- Convert pixel observations to bearing vectors -------------------------
    // processing should start from here (u,v) coordiantes are to be selected from image
    let b1 = pixel_to_bearing(u1, v1, fx, fy, cx, cy);
    let b2 = pixel_to_bearing(u2, v2, fx, fy, cx, cy);
    println!("b2 {b2:?}");
    let b3 = pixel_to_bearing(u3, v3, fx, fy, cx, cy);
    // --- Build FeatureWorldMatch tuples for lambda-twist -----------------------
    // FeatureWorldMatch<P>(bearing, WorldPoint)
    // The WorldPoint type in cv_core is a wrapper over nalgebra::Point3<f64>.
    let fw1: FeatureWorldMatch<_> = FeatureWorldMatch(b1, WorldPoint(w1.to_homogeneous()));
    let fw2: FeatureWorldMatch<_> = FeatureWorldMatch(b2, WorldPoint(w2.to_homogeneous()));
    let fw3: FeatureWorldMatch<_> = FeatureWorldMatch(b3, WorldPoint(w3.to_homogeneous()));

    // Instantiate LambdaTwist solver and estimate poses
    let solver = LambdaTwist::new();
    // It implements Estimator, so we can call estimate with an iterator of 3 FeatureWorldMatch
    let candidates = solver.estimate([fw1.clone(), fw2.clone(), fw3.clone()].into_iter());

    println!("Found {} candidate poses", candidates.len());
    let features = coords
        .into_iter()
        .zip(bearings.into_iter())
        .map(|(coords, bearing)| {
            let bearing = Unit::new_normalize(Vector3::new(bearing.x, bearing.y, 1.0));
            FeatureWorldMatch(bearing, WorldPoint(coords.to_homogeneous()))
        });

    let solver = LambdaTwist::new();
    let candidates = solver.estimate(features);

    // Reproject each candidate and compare reprojection error.
    for (idx, pose) in candidates.iter().enumerate() {
        println!("\nCandidate #{}:", idx + 1);

        for (i, world_pt) in [w1, w2, w3].iter().enumerate() {
            let proj = project_world_point_to_pixel(pose, world_pt, fx, fy, cx, cy);
            match proj {
                Some((u, v)) => {
                    println!("  pt{} -> reprojected = ({:.4}, {:.4})", i + 1, u, v);
                }
                None => {
                    println!("  pt{} -> behind camera or invalid", i + 1);
                }
            }
        }
    }

    // Optionally, pick the candidate with smallest reprojection RMS error:
    let mut best: Option<(usize, f64)> = None;
    for (idx, pose) in candidates.iter().enumerate() {
        let mut cnt = 0usize;
        let mut err_sum = 0.0f64;

        println!(
            "check matrix_world {}",
            pose.0.to_homogeneous().try_inverse().unwrap()
        );
        println!("check matrix_view {}", pose.0.to_homogeneous());

        for (u_gt, v_gt, world_pt) in &[(u1, v1, w1), (u2, v2, w2), (u3, v3, w3)] {
            if let Some((u_re, v_re)) = project_world_point_to_pixel(pose, world_pt, fx, fy, cx, cy)
            {
                let err = ((u_re - u_gt).powi(2) + (v_re - v_gt).powi(2)).sqrt();
                err_sum += err;
                cnt += 1;
            }
        }
        if cnt > 0 {
            let avg_err = err_sum / (cnt as f64);
            if best.is_none() || avg_err < best.unwrap().1 {
                best = Some((idx, avg_err));
            }
            println!(
                "Candidate {} avg reprojection error (px): {:.6}\n{}, rotion {}, {}, {}",
                idx + 1,
                avg_err,
                pose.0.translation,
                pose.0.rotation.euler_angles().0.to_degrees(),
                pose.0.rotation.euler_angles().1.to_degrees(),
                pose.0.rotation.euler_angles().2.to_degrees()
            );
        }
    }

    if let Some((best_idx, best_err)) = best {
        println!(
            "\nBest candidate is #{} with avg reprojection error {:.6} px",
            best_idx + 1,
            best_err
        );
    }
}
