import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import cv2
import sys
from collections import defaultdict
import subprocess
from typing import List, Dict, Any
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import os
import random
import logging
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import re


def extract_code_from_string(code_string, code_type="python"):
    """Extracts code or diff from a string."""
    pattern = f"```{code_type}\n([\s\S]*?)```"
    matches = re.findall(pattern, code_string, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return None



def seed_everything(seed: int, torch_deterministic=False) -> None:
    import torch
    logging.info(f"Setting seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)


def join_path(*args):
    return os.path.join(*args)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def string_to_file(string: str, filename: str) -> None:
    with open(filename, 'w') as file:
        file.write(string)


def file_to_string(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()


def create_task_config(cfg: DictConfig, task_name) -> DictConfig:
    task_config = deepcopy(cfg)
    task_config.out_dir = join_path(task_config.out_dir, task_name)
    return task_config


def load_config(config_path="../../conf", config_name="config"):
    """
    Load and merge Hydra configuration.

    :param config_path: Path to the config directory
    :param config_name: Name of the main config file (without .yaml extension)
    :return: Merged configuration object
    """
    # Initialize Hydra
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)

    # Compose the configuration
    cfg = compose(config_name=config_name)

    return cfg


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def config_to_command(cfg: DictConfig, script_path: str, conda_env: str = "articulate-anything-clean") -> List[str]:
    """
    Convert a configuration to a command-line command, flattening nested structures.

    Args:
    cfg (DictConfig): The configuration to convert.
    script_path (str): The path to the Python script to run.
    conda_env (str): The name of the Conda environment to use.

    Returns:
    List[str]: The command as a list of strings.
    """
    # Convert the configuration to a flat dictionary
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # Convert the flat dictionary to command-line arguments
    cmd_args = [f"{k}={v}" for k, v in flat_cfg.items() if v is not None]
    return make_cmd(script_path, conda_env, cmd_args)


def make_cmd(script_path: str, conda_env: str = "articulate-anything-clean",
             cmd_args=[]):
    # Construct the command
    command = [
        "conda", "run", "-n", conda_env,
        "python", script_path
    ] + cmd_args

    return command


def run_subprocess(command: List[str], env=None) -> None:
    """
    Run a command as a subprocess.

    Args:
    command (List[str]): The command to run as a list of strings.

    Raises:
    subprocess.CalledProcessError: If the command fails.
    """
    # convert all element in command to string
    command = [str(c) for c in command]
    if env is None:
        env = os.environ.copy()
    try:
        subprocess.run(command, check=True, env=env)

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e}")


class Steps:
    def __init__(self):
        self.steps = defaultdict(dict)
        self.order = []

    def add_step(self, name: str, result: Any):
        self.steps[name] = result
        self.order.append(name)

    def __getitem__(self, name):
        return self.steps[name]

    def __iter__(self):
        for name in self.order:
            yield name, self.steps[name]

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return str(self.steps)


class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def extract_frames(video_path, method="fixed", num_frames=5, interval=1):
    """
    Extract frames from a video either based on a fixed number of frames or at regular intervals.

    Parameters:
    - video_path (str): Path to the video file.
    - method (str): Method to extract frames ('fixed' or 'interval').
    - num_frames (int): Number of frames to extract (used if method is 'fixed').
    - interval (int): Interval in seconds between frames (used if method is 'interval').

    Returns:
    - frames (list): List of extracted frames.
    - frame_info (dict): Dictionary with video and frame extraction details.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    if method == "fixed":
        # Sample a fixed number of frames
        sample_indices = [int(frame_count * i / num_frames)
                          for i in range(num_frames)]
    elif method == "interval":
        # Sample frames at regular intervals
        sample_indices = [
            int(fps * i * interval) for i in range(int(duration / interval))
        ]
    else:
        raise ValueError("Invalid method. Use 'fixed' or 'interval'.")

    for idx in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()

    frame_info = {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "extracted_frame_indices": sample_indices,
    }
    frames = frames
    return frames, frame_info


def concatenate_frames_horizontally(frames):
    """
    Concatenates frames into a single image horizontally.

    Args:
        frames (list): List of PIL Images or numpy arrays to be concatenated.

    Returns:
        np.array: Concatenated image.
    """
    # Convert PIL Images to numpy arrays if necessary
    if isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]

    frames = np.array(frames)

    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(
            "The frames array must have shape (n, height, width, channels)."
        )

    concatenated_image = np.concatenate(frames, axis=1)
    return concatenated_image


def crop_white(image):
    """
    Crop white space from around a PIL image

    :param image: PIL Image object
    :return: Cropped PIL Image object
    """
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get the bounding box of the non-white area
    bg = Image.new(image.mode, image.size, (255, 255, 255))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    if bbox:
        return image.crop(bbox)
    return image  # return the original image if it's all white


def resize_frame(frame, width, height):
    if width is None and height is None:
        return frame

    original_width, original_height = frame.size

    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = original_width / original_height
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = original_height / original_width
        height = int(width * aspect_ratio)

    return frame.resize((width, height), Image.LANCZOS)


def get_frames_from_video(
    video_path,
    num_frames=5,
    video_encoding_strategy="individual",
    to_crop_white=True,
    flip_horizontal=False,
    width=None,
    height=None,

):
    frames, _ = extract_frames(video_path, num_frames=num_frames)
    pil_frames = [Image.fromarray(frame) for frame in frames]

    if flip_horizontal:
        pil_frames = [frame.transpose(Image.FLIP_LEFT_RIGHT)
                      for frame in pil_frames]

    if to_crop_white:
        pil_frames = [crop_white(frame) for frame in pil_frames]

    if width is not None or height is not None:
        # Resize the frames if either width or height is specified
        pil_frames = [resize_frame(frame, width, height)
                      for frame in pil_frames]

    if video_encoding_strategy == "concatenate":
        return [Image.fromarray(concatenate_frames_horizontally(pil_frames))]
    elif video_encoding_strategy == "individual":
        return pil_frames
    else:
        raise ValueError(
            "Invalid video_encoding_strategy. Use 'concatenate' or 'individual'."
        )


def convert_mp4_to_gif(input_path, output_path, start_time=0, end_time=None, resize=None, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return output_path

    with HideOutput():
        # Load the video file
        clip = VideoFileClip(input_path)
        # .subclip(start_time, end_time)

        # Resize if needed
        if resize:
            clip = clip.resize(resize)

        # Attempt a simpler write_gif call
        clip.write_gif(output_path, fps=10)
    return output_path


def display_frames(
    frames,
    titles=None,
    cols=5,
    figsize=(20, 10),
    border_color=None,
    border_width=20,
    wspace=0.0,
    hspace=0.0,
    save_file=None,
):
    """
    Display a list of frames with optional titles and optional colored borders.

    Parameters:
    - frames (list): List of frames to display.
    - titles (list): Optional list of titles for each frame.
    - cols (int): Number of columns in the display grid.
    - figsize (tuple): Size of the figure.
    - border_color (str): Optional color for the border around each frame.
    - border_width (int): Width of the border around each frame.
    - wspace (float): Width space between subplots.
    - hspace (float): Height space between subplots.
    """
    num_frames = len(frames)
    # Calculate the number of rows needed
    rows = (num_frames + cols - 1) // cols

    if border_color:
        frames = [draw_frame(frame, border_color, border_width)
                  for frame in frames]

    plt.figure(figsize=figsize)
    plt.ioff()
    for i, frame in enumerate(frames):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(frame)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()  # Close the figure after saving
    else:
        plt.show()


def show_video(video, overwrite=True, use_gif=False,
               num_frames=5, flip_horizontal=False):
    from IPython.display import Video, Image
    if use_gif:
        gif = convert_mp4_to_gif(video, video.replace(".mp4", ".gif"),
                                 overwrite=overwrite)
        display(Image(gif))
    else:
        frames = get_frames_from_video(video, to_crop_white=True,
                                       num_frames=num_frames,
                                       flip_horizontal=flip_horizontal)
        display_frames(frames, cols=5)