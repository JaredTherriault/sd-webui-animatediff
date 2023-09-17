import os
import gc
import cv2
import numpy as np

from typing import List, Tuple
from PIL import Image, ImageChops

from typing import Generator, Iterable, List, Optional

from scipy.interpolate import interp1d

def parse_custom_arguments(parseable_arguments):
    by_lines = parseable_arguments.split("\n")
    output = {}
    for line in by_lines:
        if line.startswith("#"):
            continue
        split = parseable_arguments.split("=")
        if len(split) == 2:
            output[split[0].strip()] = split[1].strip()
    return output
    
def interpolate_frames(images, frame_multiplier = 2, custom_arguments = "", method = "Optical Flow", *args):    
    parsed_arguments = parse_custom_arguments(custom_arguments)
    if method != "":
        output_length = len(images) * frame_multiplier
        for insertion_index in range(0, output_length, frame_multiplier):
            if insertion_index < len(images):
                number_of_frames_to_insert = frame_multiplier - 1
                keyframe_one = np.asarray(images[insertion_index - 1])
                keyframe_two = np.asarray(images[insertion_index])
    
                if method == "Optical Flow":
                    generated_frames = generate_frames_in_between(
                        keyframe_one, keyframe_two, number_of_frames_to_insert, parsed_arguments, *args)
        
                # Insert generated frames at the insertion index
                images[insertion_index:insertion_index] = generated_frames
                
    return images

def blend_image(blend_type, parsed_arguments, frame_1, frame_2, frame_num = 0):
    if blend_type == "Image.blend":
        frame_2 = np.asarray(
            Image.blend(Image.fromarray(frame_1, mode='RGB'), Image.fromarray(frame_2, mode='RGB'), float(parsed_arguments["Image.blend"])))
    elif blend_type == "cv2.addWeighted":
        frame_2 = cv2.addWeighted(
            frame_1, 1 - float(parsed_arguments["cv2.addWeighted"]), frame_2, float(parsed_arguments["cv2.addWeighted"]), 0)
    elif blend_type == "interp1d":
        t = np.linspace(0, 1, 2)
        interpolator = interp1d(
            t, [frame_1, frame_2], axis=0, 
            kind=parsed_arguments["interp1d_kind"] if "interp1d_kind" in parsed_arguments else "linear", 
            fill_value=parsed_arguments["interp1d_fill_value"] if "interp1d_fill_value" in parsed_arguments else "extrapolate")
        frame_2 = interpolator(float(parsed_arguments["interp1d"])).astype(np.uint8)
    elif blend_type == "CubicSpline":
        t = np.linspace(0, 1, 2)
        spline = CubicSpline(t, frame_arrays, axis=0)
        frame_2 = spline(float(parsed_arguments["CubicSpline"]))
    elif blend_type == "ImageChops.blend":
        frame_2 = np.asarray(
            ImageChops.blend(
                Image.fromarray(frame_1, mode='RGB'), 
                Image.fromarray(frame_2, mode='RGB'), 
                frame_num * float(parsed_arguments["ImageChops.blend"])))
    
    return Image.fromarray(frame_2, mode='RGB')
    
def generate_frames_in_between(frame_1: np.ndarray, frame_2: np.ndarray, num_frames = 1, parsed_arguments = {}, *args) -> List[np.ndarray]:
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
        
        for key in parsed_arguments.keys(): 
            # Doing a for loop here allows the blends to be ordered by the way they're ordered in the text box 
            # in case the end user wants multiple blends
            interpolated_frame = blend_image(
                key, parsed_arguments, frame_1 if frame_num == 1 else np.asarray(resultant_frames[-1]), interpolated_frame)
        
        resultant_frames.append(interpolated_frame)
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
        