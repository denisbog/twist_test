use cv::Estimator;
use cv::FeatureWorldMatch;
use cv::WorldPoint; // FeatureWorldMatch<P> and WorldPoint
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
    let fx = 800.0;
    let fy = 800.0;
    let cx = 320.0;
    let cy = 240.0;

    // Define three non-collinear world points (in meters).
    let w1 = Point3::new(0.0, 0.0, 0.0);
    let w2 = Point3::new(1.0, 0.0, 0.0);
    let w3 = Point3::new(0.0, 1.0, 0.0);

    // Define a "ground truth" camera pose (camera looking down + some translation)
    // We'll build a rotation (around X axis by -30 deg) and a translation.
    let ang = -30f64.to_radians();
    let rot = cv::nalgebra::Rotation3::from_euler_angles(ang, 0.0, 0.0);
    let trans = cv::nalgebra::Translation3::new(0.0, 0.0, 3.0); // camera is 3m in front of origin
    let gt_iso = cv::nalgebra::Isometry3::from_parts(trans, rot.into());

    // Project world points with the ground-truth pose to produce pixel observations:
    let project_gt = |p: &Point3<f64>| {
        let cp = gt_iso.transform_point(p);
        (fx * (cp.x / cp.z) + cx, fy * (cp.y / cp.z) + cy)
    };

    let (u1, v1) = project_gt(&w1);
    let (u2, v2) = project_gt(&w2);
    let (u3, v3) = project_gt(&w3);

    println!("Original pixel observations:");
    println!("p1 = ({:.3}, {:.3})", u1, v1);
    println!("p2 = ({:.3}, {:.3})", u2, v2);
    println!("p3 = ({:.3}, {:.3})", u3, v3);

    // --- Convert pixel observations to bearing vectors -------------------------
    let b1 = pixel_to_bearing(u1, v1, fx, fy, cx, cy);
    let b2 = pixel_to_bearing(u2, v2, fx, fy, cx, cy);
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
                "Candidate {} avg reprojection error (px): {:.6}",
                idx + 1,
                avg_err
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
