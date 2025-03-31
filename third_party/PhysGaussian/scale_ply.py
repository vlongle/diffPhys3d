import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def scale_gaussian_ply(input_ply, output_ply, scale_factor=0.05):
    """
    Scales the x, y, z positions of a Gaussian Splat .ply while preserving
    custom fields like f_dc_0, etc. using the plyfile library.
    
    Args:
        input_ply  (str): Path to the input .ply.
        output_ply (str): Path to the scaled output .ply.
        scale_factor (float): Scale factor to apply to x, y, z coordinates.
    """

    # 1) Read the original PLY
    plydata = PlyData.read(input_ply)

    # Typically, the GS .ply has a "vertex" element with special fields.
    vertex_data = plydata["vertex"].data  # structured array
    
    # 2) Convert to a regular ndarray so we can manipulate "x", "y", "z"
    vertex_array = np.asarray(vertex_data)  # shape: (N, ) with named fields

    # 3) Scale the positions
    # Note: check the actual names if they're spelled "x"/"y"/"z" or something else
    x_scaled = vertex_array["x"] * scale_factor
    y_scaled = vertex_array["y"] * scale_factor
    z_scaled = vertex_array["z"] * scale_factor

    # 4) Create a new structured array with the same dtype, but scaled positions
    new_vertices = np.copy(vertex_array)
    new_vertices["x"] = x_scaled
    new_vertices["y"] = y_scaled
    new_vertices["z"] = z_scaled

    # 5) Pack the updated data into a PlyElement
    # Keep the same property definitions
    el = PlyElement.describe(new_vertices, "vertex")

    # 6) Write out a new PlyData
    scaled_plydata = PlyData([el], text=plydata.text)
    scaled_plydata.write(output_ply)

    print(f"Scaled PLY saved to: {output_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to input .ply with Gaussian Splat fields")
    parser.add_argument("--output", required=True, help="Path to output scaled .ply")
    parser.add_argument("--scale",  type=float, default=0.05, help="Scale factor for x,y,z [0.05 default]")
    args = parser.parse_args()

    scale_gaussian_ply(args.input, args.output, args.scale) 