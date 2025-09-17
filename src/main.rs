use cv::Estimator;
use cv::FeatureWorldMatch;
use cv::WorldPoint;
use cv::consensus::Arrsac;
use cv::nalgebra::Isometry3;
use cv::nalgebra::Matrix3;
use cv::nalgebra::Perspective3;
use cv::nalgebra::Point2;
use cv::nalgebra::Rotation3;
use cv::nalgebra::Translation3;
use cv::nalgebra::Vector2;
use cv::nalgebra::{Point3, Unit, Vector3};
use lambda_twist::LambdaTwist;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use twist_test::pixel_to_bearing;

fn main() {
    // --- Example synthetic data -------------------------------------------------
    // Choose simple camera intrinsics:
    let fx = 4000.0;
    let fy = 3000.0;
    let cx = 2000.0;
    let cy = 1500.0;

    let projection = Perspective3::new(1.0, 100f64.to_radians(), 0.1, 1000.0);
    let projection = projection.into_inner();

    #[rustfmt::skip]
    let coords = vec![
        Point3::new(7.54, 0.0,  0.0),
        Point3::new(5.82, 0.0,  0.85),
        Point3::new(7.54, 3.77, 0.0),
        Point3::new(7.54, 0.0,  2.75),
        Point3::new(6.7, 0.0,  0.0),
        Point3::new(3.14,0.0,  2.4),
        // Point3::new(3.57,3.62, 2.75),
    ];

    // given_view_solve_for_exact_2d_pixels(fx, fy, cx, cy, &coords);

    let pixels = vec![
        Point2::new(2124.0, 1803.0),
        Point2::new(2329.0, 1668.0),
        Point2::new(1252.0, 1824.0),
        Point2::new(2152.0, 1223.0),
        Point2::new(2210.0, 1843.0),
        Point2::new(3017.0, 1102.0),
        // Point2::new(1219.0, 670.0),
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

    solve(&coords, &bearings);
}

fn given_view_solve_for_exact_2d_pixels(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    coords: &Vec<cv::nalgebra::Point<f64, cv::nalgebra::U3>>,
) {
    let trans = Translation3::new(0.536798, 3.3126, 1.31053);
    let rot = Rotation3::from_euler_angles(
        89.3215f64.to_radians(),
        -2.71232f64.to_radians(),
        -110.38f64.to_radians(),
    );
    let global_transform = Isometry3::from_parts(trans, rot.into());
    println!("view {:?}", global_transform);
    let project_gt = |p: &Point3<f64>| {
        let cp = global_transform.inverse_transform_point(p); //moving the camera
        (fx * (cp.x / cp.z) + cx, fy * (cp.y / cp.z) + cy)
    };
    println!("----- calculate with exact bearing -----");
    let bearings = coords
        .iter()
        .map(project_gt)
        .map(|item| pixel_to_bearing(item.0, item.1, fx, fy, cx, cy))
        .collect();
    solve(coords, &bearings);
}

fn solve(
    coords: &Vec<cv::nalgebra::Point<f64, cv::nalgebra::U3>>,
    bearings: &Vec<cv::nalgebra::Point<f64, cv::nalgebra::U3>>,
) {
    let features: Vec<FeatureWorldMatch<_>> = coords
        .into_iter()
        .zip(bearings.into_iter())
        .map(|(coords, bearing)| {
            let bearing = Unit::new_normalize(Vector3::new(bearing.x, bearing.y, 1.0));
            FeatureWorldMatch(bearing, WorldPoint(coords.to_homogeneous()))
        })
        .collect();

    println!("------ Find solution ------");
    let solver = LambdaTwist::new();
    let candidates = solver.estimate(features.iter().cloned());
    println!("Found {} candidate poses", candidates.len());

    // Optionally, pick the candidate with smallest reprojection RMS error:
    for (idx, pose) in candidates.iter().enumerate() {
        println!(
            "check matrix_world {}",
            pose.0.to_homogeneous().try_inverse().unwrap()
        );

        // println!("check matrix_view {}", pose.0.to_homogeneous());

        let world_matrix = pose.0.inverse();
        print_matrix(idx, world_matrix);
    }
    println!("------ Arrsac solution ------");
    use cv::Consensus;
    // Estimate potential poses with P3P.
    // Arrsac should use the fourth point to filter and find only one model from the 4 generated.
    let mut arrsac = Arrsac::new(1e-6, SmallRng::seed_from_u64(0));
    if let Some(pose) = arrsac.model(&LambdaTwist::new(), features.into_iter()) {
        let world_matrix = pose.0.inverse();

        print_matrix(0, world_matrix);
    } else {
        println!("no solution found");
    }
}

fn print_matrix(
    idx: usize,
    world_matrix: cv::nalgebra::Isometry<
        f64,
        cv::nalgebra::U3,
        cv::nalgebra::Rotation<f64, cv::nalgebra::U3>,
    >,
) {
    println!(
        "Candidate {} translation {}, rotaion {}, {}, {}",
        idx + 1,
        world_matrix.translation,
        world_matrix.rotation.euler_angles().0.to_degrees(),
        world_matrix.rotation.euler_angles().1.to_degrees(),
        world_matrix.rotation.euler_angles().2.to_degrees()
    );

    println!(
        "determinant {}",
        world_matrix.rotation.matrix().determinant()
    );
}
