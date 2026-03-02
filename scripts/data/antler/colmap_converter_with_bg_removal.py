import numpy as np
import os
from PIL import Image
import json
import rembg
import cv2
from sklearn.cluster import DBSCAN
import argparse
import matplotlib.pyplot as plt

def qvec_to_rotmat(qvec):
    """Convert COLMAP to rotation matrix"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)

def parse_cameras_txt(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cameras

def intrinsics_from_colmap_camera(cam):
    model = cam["model"].upper()
    w, h = cam["width"], cam["height"]
    p = cam["params"]

    if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = p[0], p[1], p[2]
        fx, fy = f, f
    elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"]:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(f"Unsupported/unknown COLMAP camera model: {model}")

    intr = np.array([
        [fx, fy],
        [cx, cy],
        [w,  h],
    ], dtype=np.float64)
    return intr

def parse_images_txt(path):
    images = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 10:
                image_id = int(parts[0])
                qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
                tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
                cam_id = int(parts[8])
                name = parts[9]
                images.append(dict(image_id=image_id, qvec=qvec, tvec=tvec, cam_id=cam_id, name=name))
    return images

def parse_points3D_txt(path):
    """Parse COLMAP points3D.txt to get 3D point cloud"""
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                error = float(parts[7])
                points.append({
                    'id': point_id,
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error
                })
    return points

def get_main_cluster_bounds(points3d_path, eps=0.15, min_samples=20):
    """
    Use DBSCAN clustering to find the main object cluster
    Returns bounding box and center of main cluster
    """
    print(f"Reading 3D points from {points3d_path}...")
    points = parse_points3D_txt(points3d_path)
    
    if len(points) == 0:
        print("No 3D points found")
        return None
    
    print(f"Found {len(points)} 3D points")
    
    # Extract XYZ coordinates
    xyz = np.array([p['xyz'] for p in points])
    errors = np.array([p['error'] for p in points])
    
    # Filter out points with high reconstruction error
    error_threshold = np.percentile(errors, 60)
    valid_mask = errors <= error_threshold
    xyz_filtered = xyz[valid_mask]
    
    print(f"  Filtered to {len(xyz_filtered)} points (error < {error_threshold:.2f})")
    
    # Cluster the points
    print(f"  Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_filtered)
    labels = clustering.labels_
    
    # Find largest cluster
    unique_labels = [l for l in set(labels) if l != -1]
    if len(unique_labels) == 0:
        print("  No clusters found, using all points")
        main_cluster_points = xyz_filtered
    else:
        cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        main_cluster_points = xyz_filtered[labels == largest_cluster]
        print(f"  Main cluster has {len(main_cluster_points)} points ({len(main_cluster_points)/len(xyz_filtered)*100:.1f}%)")
    
    # Calculate bounding box
    bbox_min = main_cluster_points.min(axis=0)
    bbox_max = main_cluster_points.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    
    return {
        'center': bbox_center,
        'min': bbox_min,
        'max': bbox_max,
        'size': bbox_size,
        'num_points': len(main_cluster_points),
        'all_points': main_cluster_points
    }

def project_3d_points_to_2d(points_3d, camera_pose, intrinsics):
    """
    Project multiple 3D points to 2D image coordinates
    Returns list of (x, y) coordinates that are in frame
    """
    R_c2w = camera_pose[:, :3]
    t_c2w = camera_pose[:, 3]
    
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    fx, fy = intrinsics[0, 0], intrinsics[0, 1]
    cx, cy = intrinsics[1, 0], intrinsics[1, 1]
    w, h = intrinsics[2, 0], intrinsics[2, 1]
    
    projected_points = []
    
    for point_3d in points_3d:
        # Transform to camera coordinates
        point_cam = R_w2c @ point_3d + t_w2c
        
        # Check if in front of camera
        if point_cam[2] <= 0:
            continue
        
        # Project to image
        x = (point_cam[0] / point_cam[2]) * fx + cx
        y = (point_cam[1] / point_cam[2]) * fy + cy
        
        # Check if within bounds
        if 0 <= x < w and 0 <= y < h:
            projected_points.append((int(x), int(y)))
    
    return projected_points

def create_antler_mask_from_colmap(image_shape, projected_points, point_radius=45):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(projected_points) == 0:
        return np.ones((h, w), dtype=np.uint8) * 255
    
    for px, py in projected_points:
        cv2.circle(mask, (px, py), point_radius, 255, -1)
    
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


# Outlier Detection
def calculate_mask_area(alpha_mask):
    # Convert to binary
    binary = (alpha_mask > 127).astype(np.uint8)
    
    # Count white pixels
    area = np.sum(binary)
    
    # Calculate percentage
    total = alpha_mask.shape[0] * alpha_mask.shape[1]
    percentage = (area / total) * 100
    
    return area, percentage

def detect_outliers_gaussian(areas, sigma=2.5):
    areas_array = np.array(areas)
    
    # Calculate statistics
    mean = np.mean(areas_array)
    std = np.std(areas_array)
    median = np.median(areas_array)
    
    # Calculate Z-scores
    z_scores = np.abs((areas_array - mean) / std)
    
    # Find outliers
    outlier_mask = z_scores > sigma
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    stats = {
        'mean': mean,
        'std': std,
        'median': median,
        'min': np.min(areas_array),
        'max': np.max(areas_array),
        'z_scores': z_scores
    }
    
    return outlier_indices, stats

def plot_area_analysis(areas, outliers, stats, save_path):
    """Create visualization showing area distribution and outliers"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram of areas
    ax1.hist(areas, bins=30, alpha=0.7, color='blue', edgecolor='black')
    if outliers:
        outlier_areas = [areas[i] for i in outliers]
        ax1.scatter(outlier_areas, [0]*len(outlier_areas), 
                   color='red', s=100, marker='^', label=f'Outliers (n={len(outliers)})', zorder=5)
    ax1.axvline(stats['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
    ax1.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.0f}")
    ax1.set_xlabel('Alpha Mask Area (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Alpha Mask Areas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Area per image with outliers marked plot
    indices = list(range(len(areas)))
    colors = ['red' if i in outliers else 'blue' for i in indices]
    ax2.scatter(indices, areas, c=colors, alpha=0.6)
    ax2.axhline(stats['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean")
    ax2.fill_between(indices, 
                     stats['mean'] - 2*stats['std'], 
                     stats['mean'] + 2*stats['std'],
                     alpha=0.2, color='green', label='±2σ (95% CI)')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Alpha Mask Area (pixels)')
    ax2.set_title('Alpha Mask Area per Image (Red = Outlier)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved analysis plot: {save_path}")

def colmap_to_openlrm(cameras_txt, images_txt, pose_convention="c2w"):
    """Convert COLMAP format to OpenLRM format"""
    cams = parse_cameras_txt(cameras_txt)
    imgs = parse_images_txt(images_txt)

    first_cam = cams[imgs[0]["cam_id"]]
    intrinsics = intrinsics_from_colmap_camera(first_cam)

    poses = []
    names = []
    for im in imgs:
        R_wc = qvec_to_rotmat(im["qvec"])
        t_wc = im["tvec"].reshape(3, 1)

        if pose_convention.lower() == "w2c":
            R, t = R_wc, t_wc
        elif pose_convention.lower() == "c2w":
            R = R_wc.T
            t = -R @ t_wc
        else:
            raise ValueError("pose_convention must be 'w2c' or 'c2w'")

        P = np.concatenate([R, t], axis=1)
        poses.append(P)
        names.append(im["name"])

    poses = np.stack(poses, axis=0)
    return poses, intrinsics, names

def process_single_antler(antler_id, working_dir, view_dir, use_colmap_guidance=True, save_debug=False):
    """Process a single antler with COLMAP-guided background removal and outlier detection"""
    print(f"\n{'='*70}")
    print(f"Processing {antler_id}")
    print(f"{'='*70}")
    
    # Setup paths
    antler_dir = os.path.join(view_dir, antler_id)
    raw_images_src = os.path.join(working_dir, antler_id, "setting_1", "images")
    colmap_text_path = os.path.join(working_dir, antler_id, "setting_1", "colmap", "sparse", "text")
    
    # Check paths exist
    if not os.path.exists(raw_images_src):
        print(f" Error: Images directory not found: {raw_images_src}")
        return None
    
    if not os.path.exists(colmap_text_path):
        print(f" Error: COLMAP directory not found: {colmap_text_path}")
        return None
    
    # Create output directories
    os.makedirs(antler_dir, exist_ok=True)
    os.makedirs(os.path.join(antler_dir, "pose"), exist_ok=True)
    os.makedirs(os.path.join(antler_dir, "rgba"), exist_ok=True)
    
    if save_debug:
        os.makedirs(os.path.join(antler_dir, "debug"), exist_ok=True)
    
    # Load COLMAP data
    print("\nLoading COLMAP data...")
    cameras_txt = os.path.join(colmap_text_path, "cameras.txt")
    images_txt = os.path.join(colmap_text_path, "images.txt")
    points3d_txt = os.path.join(colmap_text_path, "points3D.txt")
    
    poses, intr, names = colmap_to_openlrm(
        cameras_txt=cameras_txt,
        images_txt=images_txt,
        pose_convention="c2w"
    )
    
    print(f" Loaded {len(poses)} camera poses")
    print(f" Intrinsics shape: {intr.shape}")
    
    # Get main object cluster from COLMAP 3D points
    main_object_bbox = None
    if use_colmap_guidance and os.path.exists(points3d_txt):
        print("\nAnalyzing COLMAP 3D points to identify main object (antler)...")
        main_object_bbox = get_main_cluster_bounds(points3d_txt, eps=0.12, min_samples=15)
        
        if main_object_bbox:
            print(f"  Main object cluster identified:")
            print(f"  Center: [{main_object_bbox['center'][0]:.3f}, {main_object_bbox['center'][1]:.3f}, {main_object_bbox['center'][2]:.3f}]")
            print(f"  Size: [{main_object_bbox['size'][0]:.3f}, {main_object_bbox['size'][1]:.3f}, {main_object_bbox['size'][2]:.3f}]")
            print(f"  Points: {main_object_bbox['num_points']}")
    else:
        print("\n COLMAP guidance disabled or points3D.txt not found")
    
    # Save intrinsics and poses
    print("\nSaving camera data")
    np.save(os.path.join(antler_dir, "intrinsics.npy"), intr.astype(np.float32))
    for i, P in enumerate(poses):
        np.save(os.path.join(antler_dir, "pose", f"{i:03d}.npy"), P.astype(np.float32))
    print(f"Saved intrinsics and {len(poses)} poses")
    
    import sys
    sys.stdout.flush()
    
    try:
        rembg_session = rembg.new_session(
            model_name='birefnet-general',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    except:
        rembg_session = rembg.new_session(
            model_name='birefnet-general',
            providers=['CPUExecutionProvider']
        )
    
    sys.stdout.flush()
    
    # Process images and track areas for outlier detection
    print(f"\nProcessing {len(names)} images...")
    successful = 0
    failed = 0
    mask_areas = []
    
    for i, name in enumerate(names):
        src_path = os.path.join(raw_images_src, name)
        dst_path = os.path.join(antler_dir, "rgba", f"{i:03d}.png")
        
        print(f"\n[{i+1}/{len(names)}] {name}")
        
        # Read image
        image = cv2.imread(src_path)
        if image is None:
            print(f"Could not read image")
            failed += 1
            mask_areas.append(0)
            continue
        
        try:
            # Get initial segmentation from rembg (first pass)
            print(f"  Step 1: Initial segmentation with rembg...")
            rembg_result = rembg.remove(
                image,
                session=rembg_session,
                alpha_matting=False,
                alpha_matting_foreground_threshold=270,
                alpha_matting_background_threshold=20,
                alpha_matting_erode_size=15,
                post_process_mask=True
            )
            
            # Extract alpha channel (mask)
            if rembg_result.shape[2] == 4:
                rembg_mask_pass1 = rembg_result[:, :, 3]
            else:
                rembg_mask_pass1 = np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            # Run rembg again with more agressive settings on the result
            print(f"  Step 1 (pass 2): Second pass with aggressive settings...")
            # Create intermediate RGBA from pass 1
            temp_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            temp_rgba[:, :, 3] = rembg_mask_pass1
            
            # Convert back to BGR for second pass (rembg expects BGR)
            temp_bgr = image.copy()
            mask_inv = cv2.bitwise_not(rembg_mask_pass1)
            temp_bgr[mask_inv > 0] = [255, 255, 255]
            
            # Second rembg pass with aggressive parameters
            rembg_result_pass2 = rembg.remove(
                temp_bgr,
                session=rembg_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=285,
                alpha_matting_background_threshold=40,
                alpha_matting_erode_size=25,
                post_process_mask=True
            )
            
            if rembg_result_pass2.shape[2] == 4:
                rembg_mask = rembg_result_pass2[:, :, 3]
            else:
                rembg_mask = rembg_mask_pass1
            
            print(f"  Two-pass background removal complete")
            
            # Create COLMAP-based mask to isolate antler region
            if main_object_bbox and use_colmap_guidance:
                print(f"  Step 2: Refining with COLMAP guidance...")
                pose = poses[i]
                
                # Project all cluster points to this view
                projected_points = project_3d_points_to_2d(
                    main_object_bbox['all_points'],
                    pose,
                    intr
                )
                
                print(f"  {len(projected_points)} COLMAP points visible in this view")
                
                if len(projected_points) > 10:
                    # Create mask around projected antler points
                    colmap_mask = create_antler_mask_from_colmap(
                        image.shape,
                        projected_points,
                        expansion_ratio=1.5
                    )
                    
                    # Combine masks
                    combined_mask = cv2.bitwise_and(rembg_mask, colmap_mask)
                    
                    # Clean up noise
                    kernel = np.ones((5, 5), np.uint8)
                    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                    
                    final_mask = combined_mask
                    print(f"  Combined rembg + COLMAP masks")
                else:
                    print(f"  Not enough visible points, using rembg mask only")
                    final_mask = rembg_mask
            else:
                final_mask = rembg_mask
                print(f"  Using rembg mask only (no COLMAP guidance)")
            
            # Calculate and track area
            area, percentage = calculate_mask_area(final_mask)
            mask_areas.append(area)
            print(f"  Area: {area:,} pixels ({percentage:.1f}% of image)")
            
            # Apply final mask to create RGBA image
            rgba_result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba_result[:, :, 3] = final_mask
            
            # Save result
            cv2.imwrite(dst_path, rgba_result)
            print(f"  Saved to {dst_path}")
            
            # Save debug images
            if save_debug:
                debug_dir = os.path.join(antler_dir, "debug")
                cv2.imwrite(os.path.join(debug_dir, f"{i:03d}_original.jpg"), image)
                cv2.imwrite(os.path.join(debug_dir, f"{i:03d}_rembg_mask.png"), rembg_mask)
                if main_object_bbox and use_colmap_guidance and len(projected_points) > 10:
                    cv2.imwrite(os.path.join(debug_dir, f"{i:03d}_colmap_mask.png"), colmap_mask)
                    cv2.imwrite(os.path.join(debug_dir, f"{i:03d}_combined_mask.png"), final_mask)
                    # Draw projected points on original image
                    debug_img = image.copy()
                    for px, py in projected_points:
                        cv2.circle(debug_img, (px, py), 3, (0, 255, 0), -1)
                    cv2.imwrite(os.path.join(debug_dir, f"{i:03d}_colmap_points.jpg"), debug_img)
            
            successful += 1
            
        except Exception as e:
            print(f"  Background removal failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            mask_areas.append(0)  # Track failed as 0

    # ========================================================================
    # Outlier Detection and Analysis
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"OUTLIER DETECTION (Gaussian Method)")
    print(f"{'='*70}")
    
    # Filter out failed images (area = 0)
    valid_areas = [a for a in mask_areas if a > 0]
    valid_indices = [i for i, a in enumerate(mask_areas) if a > 0]
    
    if len(valid_areas) >= 3:
        # Detect outliers using Gaussian method with sigma=2.5
        outlier_indices_in_valid, stats = detect_outliers_gaussian(valid_areas, sigma=2.5)
        
        # Map back to original indices
        outlier_indices = [valid_indices[i] for i in outlier_indices_in_valid]
        
        print(f"\nStatistics:")
        print(f"  Mean area: {stats['mean']:,.0f} pixels")
        print(f"  Median area: {stats['median']:,.0f} pixels")
        print(f"  Std deviation: {stats['std']:,.0f} pixels")
        print(f"  Range: {stats['min']:,.0f} - {stats['max']:,.0f} pixels")
        
        if outlier_indices:
            print(f"\n⚠ Found {len(outlier_indices)} OUTLIER(S) (likely failed background removal):")
            for idx in outlier_indices:
                area = mask_areas[idx]
                ratio = area / stats['median'] if stats['median'] > 0 else 0
                z_score = stats['z_scores'][valid_indices.index(idx)]
                print(f"Image {idx:03d} ({names[idx]})")
                print(f"Area: {area:,} pixels ({ratio:.2f}x median, Z={z_score:.2f}σ)")
        else:
            print(f"\nNo outliers detected - all images have consistent foreground areas")
        
        # Create visualization
        if save_debug:
            plot_path = os.path.join(antler_dir, "debug", "outlier_analysis.png")
            plot_area_analysis(mask_areas, outlier_indices, stats, plot_path)
        
        # Save outlier report
        report_path = os.path.join(antler_dir, "outlier_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Outlier Detection Report for {antler_id}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Total images: {len(names)}\n")
            f.write(f"Valid images: {len(valid_areas)}\n")
            f.write(f"Mean area: {stats['mean']:,.0f} pixels\n")
            f.write(f"Median area: {stats['median']:,.0f} pixels\n")
            f.write(f"Std deviation: {stats['std']:,.0f} pixels\n\n")
            
            if outlier_indices:
                f.write(f"Outliers Detected: {len(outlier_indices)}\n")
                f.write(f"{'-'*50}\n")
                for idx in outlier_indices:
                    area = mask_areas[idx]
                    ratio = area / stats['median'] if stats['median'] > 0 else 0
                    z_score = stats['z_scores'][valid_indices.index(idx)]
                    f.write(f"\nImage {idx:03d}: {names[idx]}\n")
                    f.write(f"Area: {area:,} pixels\n")
                    f.write(f"Ratio to median: {ratio:.2f}x\n")
                    f.write(f"Z-score: {z_score:.2f}σ\n")
            else:
                f.write("No outliers detected.\n")
        
        print(f"\n Saved outlier report: {report_path}")
    else:
        print(f"\n Not enough valid images for outlier detection (need at least 3)")
    
    print(f"\n{'='*70}")
    print(f"Results for {antler_id}:")
    print(f"Successful: {successful}/{len(names)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(names)}")
    print(f"{'='*70}")
    
    return antler_id if successful > 0 else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare antler training data with COLMAP-guided background removal')
    parser.add_argument('--working_dir', type=str, required=True,
                       help='Base directory containing antler folders')
    parser.add_argument('--view_dir', type=str, required=True,
                       help='Output directory for training data')
    parser.add_argument('--antler_ids', type=str, nargs='+', required=True,
                       help='List of antler IDs to process (e.g., antler_6 antler_7)')
    parser.add_argument('--no_colmap_guidance', action='store_true',
                       help='Disable COLMAP-guided segmentation')
    parser.add_argument('--save_debug', action='store_true',
                       help='Save debug images showing masks and COLMAP points')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.view_dir, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"Antler Data Prep with Outlier Detection")
    print(f"{'='*70}")
    print(f"Working directory: {args.working_dir}")
    print(f"Output directory: {args.view_dir}")
    print(f"Antlers to process: {args.antler_ids}")
    print(f"COLMAP guidance: {'Disabled' if args.no_colmap_guidance else 'Enabled'}")
    print(f"Save debug images: {args.save_debug}")
    print(f"{'='*70}")
    
    # Process each antler
    processed_antlers = []
    failed_antlers = []
    
    for antler_id in args.antler_ids:
        result = process_single_antler(
            antler_id=antler_id,
            working_dir=args.working_dir,
            view_dir=args.view_dir,
            use_colmap_guidance=not args.no_colmap_guidance,
            save_debug=args.save_debug
        )
        
        if result:
            processed_antlers.append(result)
        else:
            failed_antlers.append(antler_id)
    
    # Create train/val split
    if processed_antlers:
        print(f"Creating train/validation split")
        
        split_point = max(1, int(len(processed_antlers) * 0.8))
        train_uids = processed_antlers[:split_point]
        val_uids = processed_antlers[split_point:] if split_point < len(processed_antlers) else [processed_antlers[0]]
        
        # Save JSON files
        with open(os.path.join(args.view_dir, "train_uids.json"), "w") as f:
            json.dump(train_uids, f, indent=2)
        
        with open(os.path.join(args.view_dir, "val_uids.json"), "w") as f:
            json.dump(val_uids, f, indent=2)
        
        print(f"Training samples ({len(train_uids)}): {train_uids}")
        print(f"Validation samples ({len(val_uids)}): {val_uids}")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successfully processed: {len(processed_antlers)} antlers")
        if failed_antlers:
            print(f"Failed: {len(failed_antlers)} antlers - {failed_antlers}")
        print(f"Output saved to: {args.view_dir}")
    else:
        print(f"NO ANTLERS WERE SUCCESSFULLY PROCESSED")