from PIL import Image
import numpy as np

# Adjust path as needed:
path_to_image = "data/ecb91f433f144a7798724890f0528b23/rgb_train/0001.png"
image = Image.open(path_to_image).convert("RGBA")

# Convert to NumPy array
data = np.array(image)

# For simplicity, let’s just look at the top-left corner pixel:
top_left = data[0, 0]  # (R, G, B, A) in [0..255]
print("Top-left corner pixel RGBA:", top_left)

# Or, if you suspect there's an alpha channel and you want to see
# the 'solid' background color where alpha=255, you can do:
mask_opaque = (data[:, :, 3] == 255)  # all fully opaque pixels
opaque_pixels = data[mask_opaque]
unique_opaque_colors = np.unique(opaque_pixels.reshape(-1, 4), axis=0)
print("Unique fully opaque RGBA values:", unique_opaque_colors)

# If you know the background is uniform, you could also sample
# a larger “border” region (edges of the image) to confirm all match.
