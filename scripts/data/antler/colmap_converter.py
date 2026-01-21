import numpy as np
import os
from PIL import Image
import json
import rembg
import cv2

def qvec_to_rotmat(qvec):
    # COLMAP qvec: [qw, qx, qy, qz]
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
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
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
    with open(path, "r", encoding="gb2312") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # First line of each image block has 10 fields min:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
            if len(parts) == 10:
                image_id = int(parts[0])
                qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
                tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
                cam_id = int(parts[8])
                name = parts[9]
                images.append(dict(image_id=image_id, qvec=qvec, tvec=tvec, cam_id=cam_id, name=name))
            else:
                # second line in the block = 2D-3D correspondences, skip
                pass
    return images

def colmap_to_openlrm(cameras_txt, images_txt, pose_convention="w2c"):
    """
    pose_convention:
      - "w2c": poses are world->camera (COLMAP native)  x_cam = R X_world + t
      - "c2w": poses are camera->world                 X_world = R x_cam + t
    """
    cams = parse_cameras_txt(cameras_txt)
    imgs = parse_images_txt(images_txt)

    # If you have multiple camera IDs, you may need per-image intrinsics.
    # Here we assume a single camera, or that you only need one intrinsics array.
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

        P = np.concatenate([R, t], axis=1)  # [3,4]
        poses.append(P)
        names.append(im["name"])

    poses = np.stack(poses, axis=0)  # [M,3,4]
    return poses, intrinsics, names

if __name__ == "__main__":
    working_dir = "/mnt/home/lihao/lihao_project/antler_colmap"
    view_dir = f"{working_dir}/views"
    os.makedirs(view_dir, exist_ok=True)
    antler_id = "antler_test"
    antler_dir = os.path.join(view_dir, antler_id)
    os.makedirs(antler_dir, exist_ok=True)
    poses, intr, names = colmap_to_openlrm(
        cameras_txt=f"{working_dir}/sparse/text/cameras.txt",
        images_txt=f"{working_dir}/sparse/text/images.txt",
        pose_convention="c2w"  # c2w is used by OpenLRM
    )
    
    # Visualize poses
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Extract camera positions from poses (c2w convention: position is last column of inverse)
    # positions = []
    # for P in poses:
    #     R = P[:, :3]
    #     t = P[:, 3]
    #     # For c2w: camera position is -R^T @ t
    #     cam_pos = -R.T @ t
    #     positions.append(cam_pos)

    # positions = np.array(positions)

    # # Plot camera positions
    # ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=50, label='Camera positions')

    # # Plot camera frustums (simple arrows pointing in viewing direction)
    # for i, P in enumerate(poses):
    #     R = P[:, :3]
    #     pos = positions[i]
    #     # Camera z-axis in world frame (viewing direction for c2w)
    #     direction = R.T[:, 2] * 0.1
    #     ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], color='blue', arrow_length_ratio=0.3, length=1)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # ax.set_title('Camera Poses Visualization')
    # plt.savefig('poses_visualization.png', dpi=150, bbox_inches='tight')
    # print(f"Saved visualization to 'poses_visualization.png'")

    print(poses.shape)  # (M, 3, 4)
    print(intr.shape)   # (3, 2)
    print(intr)
    exit()
    # store intr as binary float32 npy
    np.save(os.path.join(antler_dir, "intrinsics.npy"), intr.astype(np.float32))
    os.makedirs(os.path.join(antler_dir, "pose"), exist_ok=True)
    os.makedirs(os.path.join(antler_dir, "rgba"), exist_ok=True)
    # store each pose as binary float32 npy with name 000.npy, 001.npy, ...
    for i, P in enumerate(poses):
        np.save(os.path.join(antler_dir, "pose", f"{i:03d}.npy"), P.astype(np.float32))
    # move images to rgba/ with names 000.png, 001.png, ...
    for i, name in enumerate(names):
        print(f"Removing image {name} background and move to {i:03d}.png")
        src_path = os.path.join(working_dir, name)
        dst_path = os.path.join(antler_dir, "rgba", f"{i:03d}.png")
        image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        rembg_session = rembg.new_session(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        image_out = rembg.remove(data=image, session=rembg_session)
        cv2.imwrite(dst_path, image_out)

    
    # generate train_uids.json and val_uids.json
    num_views = poses.shape[0]
    train_uids = [antler_id]
    val_uids = [antler_id]
    with open(os.path.join(view_dir, "train_uids.json"), "w") as f:
        json.dump(train_uids, f)
    with open(os.path.join(view_dir, "val_uids.json"), "w") as f:
        json.dump(val_uids, f)