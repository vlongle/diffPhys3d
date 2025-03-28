from PIL import Image
import os
import argparse

def convert_images(obj_id):
    """
    Remove the alpha channel from images and reorganize the dataset structure.
    
    Args:
        obj_id (str): The object ID of the dataset to process
    """
    # Define paths based on the object ID
    # source_data = f"/mnt/kostas-graid/datasets/vlongle/diffphys3d/data/{obj_id}"
    source_data = f"data/{obj_id}"
    input_dir = f"{source_data}/train"
    output_dir = f"{source_data}/rgb_train"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert images from RGBA to RGB
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            # Convert to RGB if it has an alpha channel
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(os.path.join(output_dir, filename))
    
    # Rename folders to match expected structure
    os.rename(input_dir, f"{source_data}/rgba_train")
    os.rename(output_dir, f"{source_data}/train")
    
    # Rename transforms file
    if os.path.exists(f"{source_data}/transforms_train.json"):
        os.rename(f"{source_data}/transforms_train.json", f"{source_data}/transforms.json")
    
    print(f"Successfully processed dataset for object ID: {obj_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NGP format dataset for use with LERF, F3RM, etc.")
    parser.add_argument("--obj_id", type=str, help="Object ID of the dataset to process")
    args = parser.parse_args()
    
    convert_images(args.obj_id)