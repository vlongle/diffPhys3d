import os
import argparse
import glob
import os.path as osp
from vlmx.utils import save_json
from dotenv import load_dotenv
from tqdm import tqdm

def get_rendered_images(save_folder, obj_class=None):
    """
    Returns a dictionary mapping object IDs to their rendered image paths.
    
    Args:
        save_folder (str): Path to the folder containing rendered images
        obj_class (str, optional): Specific object class to filter by
        
    Returns:
        dict: Dictionary with object IDs as keys and image paths as values
    """
    result_dict = {}
    
    # Determine the base path based on whether obj_class is provided
    if obj_class:
        base_path = osp.join(save_folder, obj_class)
    else:
        base_path = save_folder
    
    # Find all category folders if no specific class is provided
    if obj_class:
        categories = [obj_class]
    else:
        categories = [osp.basename(f.rstrip('/')) for f in sorted(glob.glob(f"{base_path}/*/"))]
    
    # Iterate through all categories
    for category in categories:
        category_path = osp.join(save_folder, category)
        
        # Get all object folders within this category
        object_folders = sorted(glob.glob(f"{category_path}/*/"))
        
        for obj_folder in object_folders:
            obj_id = osp.basename(obj_folder.rstrip('/'))
            
            # Look for subfolders that contain the rendered images
            render_folders = sorted(glob.glob(f"{obj_folder}/*/"))
            
            for render_folder in render_folders:
                # Look for PNG files (typically 000.png for the first view)
                png_files = sorted(glob.glob(f"{render_folder}/*.png"))
                
                if png_files:
                    # Use the first image found (typically 000.png)
                    image_path = png_files[0]
                    
                    # Create a unique ID that includes category and object ID
                    unique_id = f"{category}/{obj_id}/{osp.basename(render_folder.rstrip('/'))}"
                    result_dict[unique_id] = image_path
    
    return result_dict

def process_images_with_vlm(rendered_images, class_name, api_key, output_dir="vlm_results"):
    """
    Process rendered images with a VLM and save the results.
    
    Args:
        rendered_images (dict): Dictionary mapping object IDs to image paths
        class_name (str): Class name to use in the VLM prompt
        api_key (str): API key for the VLM service
        output_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary with object IDs and their appropriateness status
    """
    from vlmx.agent import Agent, AgentConfig
    from PIL import Image
    import json
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    SYSTEM_INSTRUCTION = f"""
    We need to select some images of the classes: {class_name}. We will provide you some image rendered from the 3d model. You need to either return True or False. 
    Return False to reject the image as inappropriate for the video game development. Some common reasons for rejection:
    - The image doesn't clearly depict the object class
    - The image contains other things in addition to the object. REMEMBER, we only want images that depict ONE SINGLE OBJECT belong to one of the classes.

    The return format is
    ```json
    {{
    "is_appropriate": true (or false),
    "reason": "reason for the decision"
    }}
    ```
    """

    ADDITIONAL_INSTRUCTION = """
    We'll be using the 3d models to learn physic parameters like material and young modulus to simulate the physics of the object.
    E.g., the tree swaying in the wind. Therefore, you need to decide if the image depicts an object that is likely to be used in a physics simulation.
    """
    
    class HelperAgent(Agent):
        OUT_RESULT_PATH = "vlm_results.json"
        def _make_system_instruction(self):
            return SYSTEM_INSTRUCTION + ADDITIONAL_INSTRUCTION

        def _make_prompt_parts(self, image_path: str):
            question = ["The image is :", Image.open(image_path)]
            return question

        def parse_response(self, response):
            json_str = response.text.strip().strip("```json").strip()
            parsed_response = json.loads(json_str, strict=False)
            save_json(parsed_response, os.path.join(
            self.cfg.out_dir, self.OUT_RESULT_PATH))
            return parsed_response

    
    # Initialize the agent
    
    results = {}
    
    # Process each image with a progress bar
    for obj_id, image_path in tqdm(rendered_images.items(), desc="Processing images", unit="image"):
        try:
            agent = HelperAgent(AgentConfig(
                # model_name="gemini-2.0-flash-thinking-exp-01-21",
                model_name="gemini-1.5-flash-latest",
                out_dir=os.path.join(output_dir, obj_id),
                api_key=api_key
            ))

            agent.generate_prediction(image_path)
            response = agent.load_prediction()
            results[obj_id] = response 

        except Exception as e:
            print(f"Error processing {obj_id}: {e}")
            results[obj_id] = {"is_appropriate": False, "reason": f"Processing error: {str(e)}"}
        
    print("results", results)
    save_json(results, os.path.join(output_dir, "all_results.json"))
    
    return results

def analyze_vlm_results(results):
    """
    Analyze the results from the VLM processing.
    
    Args:
        results (dict): Dictionary with object IDs and their appropriateness status
        
    Returns:
        tuple: (total_objects, appropriate_objects, inappropriate_objects, stats_by_category)
    """
    total_objects = len(results)
    appropriate_objects = sum(1 for result in results.values() if result.get("is_appropriate", False))
    inappropriate_objects = total_objects - appropriate_objects
    
    # Analyze by category
    stats_by_category = {}
    for obj_id, result in results.items():
        category = obj_id.split('/')[0]
        if category not in stats_by_category:
            stats_by_category[category] = {
                "total": 0,
                "appropriate": 0,
                "inappropriate": 0
            }
        
        stats_by_category[category]["total"] += 1
        if result.get("is_appropriate", False):
            stats_by_category[category]["appropriate"] += 1
        else:
            stats_by_category[category]["inappropriate"] += 1
    
    return total_objects, appropriate_objects, inappropriate_objects, stats_by_category

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process rendered images with VLM')
    parser.add_argument(
        '--save_folder', type=str, default='class_render_outputs',
        help='path to folder containing rendered images')
    parser.add_argument(
        '--obj_class', type=str, default="tree",
        help='object class to filter by (e.g., "mug")')
    parser.add_argument(
        '--class_name', type=str, default="tree, ficus, fern",
        help='class name to use in the VLM prompt')
    parser.add_argument(
        '--output_dir', type=str, default="vlm_results",
        help='directory to save VLM results')
    parser.add_argument(
        '--analyze_only', action='store_true',
        help='only analyze existing results without running VLM')
    
    args = parser.parse_args()
    
    # Get the dictionary of rendered images
    rendered_images = get_rendered_images(args.save_folder, args.obj_class)
    print(f"Found {len(rendered_images)} rendered images")
    
    
    load_dotenv()
    API_KEY = os.environ.get('API_KEY')

    if not args.analyze_only:
        # Process images with VLM
        results = process_images_with_vlm(
            rendered_images, 
            args.class_name, 
            API_KEY, 
            args.output_dir
        )
    else:
        # Load existing results
        import json
        try:
            with open(os.path.join(args.output_dir, "all_results.json"), 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Results file not found at {os.path.join(args.output_dir, 'all_results.json')}")
            exit(1)
    
    # Analyze results
    total, appropriate, inappropriate, stats_by_category = analyze_vlm_results(results)
    
    # Print analysis
    print("\nVLM Analysis Results:")
    print(f"Total objects: {total}")
    print(f"Appropriate objects: {appropriate} ({appropriate/total*100:.2f}%)")
    print(f"Inappropriate objects: {inappropriate} ({inappropriate/total*100:.2f}%)")
    
    print("\nResults by category:")
    for category, stats in stats_by_category.items():
        print(f"  {category}:")
        print(f"    Total: {stats['total']}")
        print(f"    Appropriate: {stats['appropriate']} ({stats['appropriate']/stats['total']*100:.2f}%)")
        print(f"    Inappropriate: {stats['inappropriate']} ({stats['inappropriate']/stats['total']*100:.2f}%)")
