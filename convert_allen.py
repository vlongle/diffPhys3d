import json
import os
import shutil
import numpy as np
import argparse

def convert_to_ngp_format(blender_rendered_dir, target_dir):
    metadata_path = os.path.join(blender_rendered_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get camera parameters
    cam_fov_x = metadata['cam_fov_x']
    cam_focal_length_mm = metadata['cam_focal_length_mm']
    cam_sensor_width = metadata['cam_sensor_width']
    
    # Calculate focal length in pixels (for 512x512 images)
    image_width = 512
    image_height = 512
    focal_length_pixels = (cam_focal_length_mm / cam_sensor_width) * image_width

    image_paths = [os.path.join(blender_rendered_dir, fn) for fn in os.listdir(blender_rendered_dir) if fn.endswith('.png')]
    image_paths.sort()

    transform_paths = [fn.replace('.png', '.npy') for fn in image_paths]

    os.makedirs(target_dir, exist_ok=True)

    # Copy images
    img_dir = os.path.join(target_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for src_img_path in image_paths:
        dst_img_path = os.path.join(img_dir, os.path.basename(src_img_path))
        shutil.copyfile(src_img_path, dst_img_path)

    # Create transforms.json for NGP
    transforms_json_dict = {
        "aabb_scale": 4,  # Typical value for NGP
        "frames": [],
        "camera_angle_x": cam_fov_x,
        "fl_x": focal_length_pixels,
        "fl_y": focal_length_pixels,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": image_width / 2,
        "cy": image_height / 2,
        "w": image_width,
        "h": image_height,
    }

    for frame_idx, frame_path in enumerate(image_paths):
        # Load the camera matrix
        opencv2wld = np.load(transform_paths[frame_idx])  # (3, 4)
        opencv2wld = np.vstack([opencv2wld, [0, 0, 0, 1]])  # (4, 4)
        
        # NGP uses a different camera convention than Blender
        # Convert from OpenCV to OpenGL/NGP convention
        R = opencv2wld[:3, :3]
        t = opencv2wld[:3, 3]
        
        # Flip the y and z axes to convert from OpenCV to OpenGL
        flip = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply the flip transformation
        transform_matrix = flip @ opencv2wld
        
        # For NGP, we need the camera-to-world transform
        transform_matrix = np.linalg.inv(transform_matrix)

        # Get the file path relative to the target directory
        rel_path = os.path.join('images', os.path.basename(frame_path))

        per_frame_dict = {
            'file_path': rel_path,
            'transform_matrix': transform_matrix.tolist(),
            'sharpness': 50.0  # Default sharpness value
        }

        transforms_json_dict['frames'].append(per_frame_dict)

    # Save the transforms.json file
    transforms_path = os.path.join(target_dir, 'transforms.json')
    with open(transforms_path, 'w') as f:
        json.dump(transforms_json_dict, f, indent=4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_rendered_dir", type=str, help="Path to the Blender rendered directory")
    parser.add_argument("--target_dir", type=str, help="Path to the target directory")
    args = parser.parse_args()
    convert_to_ngp_format(args.blender_rendered_dir, args.target_dir)
