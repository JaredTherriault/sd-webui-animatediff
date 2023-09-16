import os
import gc
import cv2
import numpy as np

from typing import List, Tuple
from PIL import Image

from typing import Generator, Iterable, List, Optional
    
def interpolate_frames(images, start_iterating_from_image_number = 0, frame_multiplier = 2, method = "Optical Flow", *args):    
    if method != "":
        output_length = len(images) * frame_multiplier
        for insertion_index in range(start_iterating_from_image_number, output_length, frame_multiplier):
            if insertion_index < len(images):
                number_of_frames_to_insert = frame_multiplier - 1
                keyframe_one = np.asarray(images[insertion_index - 1])
                keyframe_two = np.asarray(images[insertion_index])
    
                if method == "Optical Flow":
                    generated_frames = generate_frames_in_between(keyframe_one, keyframe_two, number_of_frames_to_insert, *args)
        
                # Insert generated frames at the insertion index
                images[insertion_index:insertion_index] = generated_frames
    
def generate_frames_in_between(frame_1: np.ndarray, frame_2: np.ndarray, num_frames = 1, *args) -> List[np.ndarray]:
    """Generate intermediate frames between two frames using optical flow.

    Args:
        frame_1: The first frame as a NumPy array.
        frame_2: The second frame as a NumPy array.
        num_frames: The number of intermediate frames to generate.

    Returns:
        A list of generated frames as NumPy arrays.
    """
    def median_filter(flow, kernel_size=5):
        return cv2.medianBlur(flow, kernel_size)
    def gaussian_smooth(flow, sigma=1.5):
        return cv2.GaussianBlur(flow, (0, 0), sigma)
    resultant_frames = []
    optical_flow = calculate_optical_flow_between_frames(frame_1, frame_2, *args)
    optical_flow = median_filter(optical_flow)
    optical_flow = gaussian_smooth(optical_flow)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        border_mode = cv2.BORDER_REFLECT
        interpolated_frame = cv2.remap(frame_1, flow, None, cv2.INTER_LANCZOS4, borderMode=1) # INTER_LANCZOS4 is p good
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
        