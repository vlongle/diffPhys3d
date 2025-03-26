import objaverse
import argparse

def download_object(obj_id):
    """
    Download an object from Objaverse using its unique ID.
    
    Args:
        obj_id (str): The unique ID of the object to download
    """
    # Download the object
    objects = objaverse.load_objects(uids=[obj_id])
    print(f"Successfully downloaded object: {obj_id}")
    print("Object data:", objects)
    
    # Optionally print categories if available
    try:
        categories = objaverse.load_annotations([obj_id])[obj_id]['categories']
        print(f"Object categories: {categories}")
    except (KeyError, Exception) as e:
        print(f"Could not load categories: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an object from Objaverse")
    parser.add_argument("--obj_id", type=str, help="Unique ID of the object to download")
    args = parser.parse_args()
    
    download_object(args.obj_id)
