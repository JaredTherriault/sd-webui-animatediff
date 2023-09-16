import os
import gc
import cv2
import gradio as gr
import numpy as np

from typing import List, Tuple
from PIL import Image

# From AnimateDiff extension
from scripts.logging_animatediff import logger_animatediff

# From webui
from modules import shared

from typing import Generator, Iterable, List, Optional

def get_interpolation_ui_controls():
    with gr.Accordion('Interpolation', open=False):
        with gr.Row():
            interpolation_enabled = gr.Checkbox(True, label="Interpolation Enabled", type="value")
            frame_multiplier = gr.Number(minimum=2, value=2, precision=0, label="Frame Multipler", type="value", info="This will multiply the frame rate by the specified amount, but will not affect the video length.")

        with gr.Accordion('Advanced', open=False):
            method = gr.Dropdown(choices=["Optical Flow"], value="Optical Flow", label="Interpolation Method", type="value")
            with gr.Row():
                pyr_scale = gr.Number(minimum=0.0000001, maximum=0.9999999, value=0.5, step=0.01, label="pyr_scale", type="value", info="Scale of the image pyramid used in the optical flow calculation. Smaller value = finer pyramid = more levels = more detail but may be slower.")
                levels = gr.Number(minimum=1, value=3, precision=0, label="levels", type="value", info="Determines the number of levels in the image pyramid. More levels can capture motion at different scales but may increase computational cost.")
                winsize = gr.Number(minimum=1, value=16, precision=0, label="winsize", type="value", info="Defines the size of the pixel neighborhood used for flow integration. Larger winsize can handle larger motions but may blur small details. ")
            with gr.Row(): 
                iterations = gr.Number(minimum=1, value=3, precision=0, label="iterations", type="value", info="Controls the number of iterations at each pyramid level. More iterations can improve accuracy but increase computation time.")
                poly_n = gr.Number(minimum=1, value=5, precision=0, label="poly_n", type="value", info="Pixel neighborhood size used when estimating polynomial expansion. A higher value can capture more complex motion but may increase noise.")
                poly_sigma = gr.Number(minimum=0.0000001, value=1.5, step=0.1, label="poly_sigma", type="value", info="Standard deviation of Gaussian to smooth polynomial expansion derivatives. Larger value = stronger smoothing, which can reduce noise.")
                flags = gr.Number(minimum=0, value=0, precision=0, label="flags", type="value", info="Include additional flags to customize the optical flow calculation. The value of 0 typically means no additional flags are used.")
    return interpolation_enabled, frame_multiplier, method, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    
def interpolate_frames(res, frame_multiplier = 2, method = "Optical Flow", *args):    
    if method != "":
        output_length = len(res.images) * frame_multiplier
        for insertion_index in range(res.index_of_first_image + 1, output_length, frame_multiplier):
            if insertion_index < len(res.images):
                number_of_frames_to_insert = frame_multiplier - 1
                keyframe_one = np.asarray(res.images[insertion_index - 1])
                keyframe_two = np.asarray(res.images[insertion_index])
    
                if method == "Optical Flow":
                    generated_frames = generate_frames_in_between(keyframe_one, keyframe_two, number_of_frames_to_insert, *args)
        
                # Insert generated frames at the insertion index
                res.images[insertion_index:insertion_index] = generated_frames
    
def generate_frames_in_between(frame_1: np.ndarray, frame_2: np.ndarray, num_frames = 1, *args) -> List[np.ndarray]:
    """Generate intermediate frames between two frames using optical flow.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.
        num_frames: The number of intermediate frames to generate.

    Returns:
        A list of generated frames as NumPy arrays.
    """
    resultant_frames = []
    optical_flow = calculate_optical_flow_between_frames(frame_1, frame_2, *args)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        border_mode = cv2.BORDER_REFLECT
        interpolated_frame = cv2.remap(frame_1, flow, None, cv2.INTER_LINEAR, borderMode=1) # INTER_LANCZOS4 is p good
        resultant_frames.append(Image.fromarray(interpolated_frame, mode="RGB"))
    return resultant_frames

def calculate_optical_flow_between_frames(  frame_1: np.ndarray, 
                                            frame_2: np.ndarray, 
                                            pyr_scale = 0.5,
                                            levels = 3,
                                            winsize = 128,
                                            iterations = 3,
                                            poly_n = 5,
                                            poly_sigma = 6,
                                            flags = 0,
                                            *args) -> np.ndarray:
    """Calculate optical flow between two frames.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.

    Returns:
        The optical flow as a NumPy array.
    """
    frame_1_gray, frame_2_gray = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY ), cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY )
    optical_flow = cv2.calcOpticalFlowFarneback(frame_1_gray,
                                                frame_2_gray,
                                                None,
                                                pyr_scale = pyr_scale,
                                                levels = levels,
                                                winsize = winsize,
                                                iterations = iterations,
                                                poly_n = poly_n,
                                                poly_sigma = poly_sigma,
                                                flags = flags
                                                )
    return optical_flow
        