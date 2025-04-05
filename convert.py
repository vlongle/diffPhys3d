from PIL import Image
import os
import argparse
import shutil

def convert_images(data_dir, obj_id):
    """
    Remove the alpha channel from images and reorganize the dataset structure.
    
    Args:
        obj_id (str): The object ID of the dataset to process
    """
    # Define paths based on the object ID
    # source_data = f"/mnt/kostas-graid/datasets/vlongle/diffphys3d/data/{obj_id}"
    source_data = f"{data_dir}/{obj_id}"
    # source_data = f"feat_data/{obj_id}"
    output_dir = f"{source_data}/train"
    os.makedirs(output_dir, exist_ok=True)
    
    
    for filename in os.listdir(output_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(output_dir, filename)
            img = Image.open(img_path)
            # Convert to RGB if it has an alpha channel
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(os.path.join(output_dir, filename))
            # print("img.mode", img.mode, "saved to", os.path.join(output_dir, filename))
    
    # # Rename transforms file
    if os.path.exists(f"{source_data}/transforms_train.json"):
        # Copy instead of rename to keep both files
        shutil.copy(f"{source_data}/transforms_train.json", f"{source_data}/transforms.json")
        print(f"Copied transforms_train.json to transforms.json")
    
    # Create an empty transforms_test.json with minimal valid structure
    import json
    empty_transforms = {
        "frames": [],
        "camera_angle_x": 0.0,
        "camera_angle_y": 0.0
    }
    
    with open(f"{source_data}/transforms_test.json", 'w') as f:
        json.dump(empty_transforms, f, indent=4)
    print(f"Created empty transforms_test.json")
    
    print(f"Successfully processed dataset for object ID: {obj_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NGP format dataset for use with LERF, F3RM, etc.")
    parser.add_argument("--obj_id", type=str, help="Object ID of the dataset to process")
    parser.add_argument("--data_dir", type=str, help="Data directory", default="data")
    args = parser.parse_args()
    
    convert_images(args.data_dir, args.obj_id)