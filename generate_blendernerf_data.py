import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import bpy
from mathutils import Vector
import shutil
import json

def get_default_output_dir(format_type):
    """Determine the default output directory based on the format."""
    home_dir = os.path.expanduser("~")
    if format_type == "NGP":
        return os.path.join(home_dir, "code", "instant-ngp", "data")
    else:  # NERF format for gaussian splatting
        return os.path.join(home_dir, "code", "gaussian-splatting", "data")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default=None, 
                   help="Path to output directory. If not provided, will use format-specific default location.")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=float, default=1.5)
parser.add_argument("--format", type=str, default="NERF", choices=["NERF", "NGP"])
parser.add_argument("--only_normalize", action='store_true', help="Only normalize the scene, don't render")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

# Set the output directory if not provided
if args.output_dir is None:
    args.output_dir = get_default_output_dir(args.format)
    print(f"Using default output directory: {args.output_dir}")

context = bpy.context
scene = context.scene
render = scene.render


## configure rendering settings
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
# render.resolution_x = 800
# render.resolution_y = 800
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True


### extra settings to ensure pure white background for gaussian splatting...
# scene.world.light_settings.use_ambient_occlusion = False
scene.world.use_nodes = True
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'

## NOTE: for some reason, with transparent background, gaussian splatting will learn this weird
## artifacts like floating celling stuff surrounding the object.
scene.render.film_transparent = args.format == "NGP" 

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )



def add_lighting() -> None:
    """Add a professional studio-like lighting setup with multiple area lights."""
    # Delete the default light
    if "Light" in bpy.data.objects:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    
    # Clear any existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create a three-point lighting setup
    
    # 1. Key light (main light) - brightest, from front-right
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 2))
    key_light = bpy.context.object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 5
    key_light.rotation_euler = (0.6, 0.2, 0.8)  # Angle toward the subject
    
    # 2. Fill light - softer light from opposite side to fill shadows
    bpy.ops.object.light_add(type="AREA", location=(-2, -1, 1))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200  # Less intense than key light
    fill_light.data.size = 7  # Larger for softer light
    fill_light.rotation_euler = (0.5, -0.2, -0.8)
    
    # 3. Rim/Back light - creates separation from background
    bpy.ops.object.light_add(type="AREA", location=(0, 3, 2))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 300
    rim_light.data.size = 4
    rim_light.rotation_euler = (0.8, 0, 0)  # Point down at the back of subject
    
    # 4. Top light for general fill
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4))
    top_light = bpy.context.object
    top_light.name = "Top_Light"
    top_light.data.energy = 150
    top_light.data.size = 10
    top_light.rotation_euler = (0, 0, 0)  # Point straight down
    
    # set the background to white
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    bg_node.inputs[1].default_value = 1.0  # Full strength


# def add_lighting() -> None:
#     # delete the default light
#     bpy.data.objects["Light"].select_set(True)
#     bpy.ops.object.delete()
#     # add a new light
#     bpy.ops.object.light_add(type="AREA")
#     light2 = bpy.data.lights["Area"]
#     light2.energy = 30000
#     bpy.data.objects["Area"].location[2] = 0.5
#     bpy.data.objects["Area"].scale[0] = 100
#     bpy.data.objects["Area"].scale[1] = 100
#     bpy.data.objects["Area"].scale[2] = 100


#     # Add a world background color
#     world = bpy.data.worlds['World']
#     world.use_nodes = True
#     bg_node = world.node_tree.nodes['Background']
#     bg_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Light gray background


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)





# load the model
def load_object(object_path: str) -> None:
    """Loads a 3D model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    
    bpy.ops.object.select_all(action="DESELECT")

def setup_manual_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    return cam, cam_constraint

def render_with_blendernerf(object_uid: str) -> None:
    """Use BlenderNerf add-on to render the normalized scene."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, object_uid)
    
    # Clear the output directory if it exists to remove stale data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up parameters for the BlenderNeRF add-on
    scene = bpy.context.scene
    
    # Global parameters
    scene.train_data = True
    scene.test_data = False
    scene.aabb = 4  # Smaller bounding box to focus on the object
    scene.render_frames = True
    scene.nerf = args.format == "NERF"  # True for NeRF format, False for NGP format
    scene.save_path = output_dir
    
    # COS specific parameters
    scene.cos_dataset_name = "dataset"
    scene.sphere_location = (0.0, 0.0, 0.0)  # Centered at origin after normalization
    scene.sphere_rotation = (0.0, 0.0, 0.0)
    scene.sphere_scale = (1.0, 1.0, 1.0)
    scene.sphere_radius = args.camera_dist / 2
    scene.focal = 20.0  # lens focal length in mm
    scene.cos_nb_frames = args.num_images
    scene.seed = 0
    # scene.upper_views = False
    scene.upper_views = True
    scene.outwards = False
    
    try:
        # Run the Camera on Sphere operator from BlenderNerf
        bpy.ops.object.camera_on_sphere()
        print(f"Successfully rendered {args.num_images} images using BlenderNerf add-on")
    except Exception as e:
        pass


    shutil.unpack_archive( output_dir + "/dataset.zip", output_dir)
    os.remove(output_dir + "/dataset.zip")

    if args.format == "NERF":
        # create a dummy transforms_test.json
        with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
            json.dump({"camera_angle_x": 0.0, "frames": []}, f)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path

def process_object(object_path: str) -> None:
    """Process a single object: load, normalize, and render."""
    reset_scene()
    
    # Load the object
    load_object(object_path)
    object_uid = os.path.basename(object_path).split(".")[0]
    
    # Normalize the scene
    normalize_scene()
    print(f"Scene normalized for {object_uid}")
    
    if not args.only_normalize:
        # Add lighting
        add_lighting()
        
        # Render with BlenderNerf add-on
        render_with_blendernerf(object_uid)

if __name__ == "__main__":
    try:
        start_i = time.time()
        
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        
        process_object(local_path)
        
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        
        # Delete the object if it was downloaded
        if args.object_path.startswith("http") and not args.only_normalize:
            os.remove(local_path)
            
    except Exception as e:
        print("Failed to process", args.object_path)
        print(e)

