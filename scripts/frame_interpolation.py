import os
import gc
import cv2
import copy
import numpy as np

from typing import List, Tuple
from PIL import Image, ImageChops

from typing import Generator, Iterable, List, Optional

from scipy.interpolate import interp1d, CubicSpline

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def parse_custom_arguments(custom_arguments):
    if "\n" in custom_arguments:
        by_lines = custom_arguments.split("\n")
    elif "," in custom_arguments:
        by_lines = custom_arguments.split(",")
    else:
        return {}
    
    output = {}
    for line in by_lines:
        if line.startswith("#") or line.strip() == "": # Skip lines that are blank or commented out
            continue
        split = line.split("=")
        if len(split) > 1:
            output[split[0].strip()] = split[1].strip()
        else: # sometimes we just have a flag with no "="
            output[line] = line
    return output
    
def interpolate_frames(images, frame_multiplier = 2, custom_arguments = "", method = "Optical Flow", *args):    
    parsed_arguments = parse_custom_arguments(custom_arguments)
    output_images = []
    if method != "":
        for original_frame_number in range(len(images) - 1): # -1 because we don't need to interpolate after the final frame
            
            output_images.append(images[original_frame_number])
            
            number_of_frames_to_insert = frame_multiplier - 1
            keyframe_one = images[original_frame_number]
            keyframe_two = images[original_frame_number + 1]

            if method == "Optical Flow":
                if "UseAltOpticalFlow1" in parsed_arguments.keys():
                    generated_frames = generate_pil_frames_in_between_alt1(
                        keyframe_one, keyframe_two, number_of_frames_to_insert, parsed_arguments, *args)
                elif "UseAltOpticalFlow2" in parsed_arguments.keys():
                    generated_frames = generate_pil_frames_in_between_alt2(
                        keyframe_one, keyframe_two, number_of_frames_to_insert, parsed_arguments, *args)
                else:
                    generated_frames = generate_pil_frames_in_between(
                        keyframe_one, keyframe_two, number_of_frames_to_insert, parsed_arguments, *args)
    
            # Append generated frames
            output_images += generated_frames
            
        output_images.append(images[-1]) # Append final frame
        
        output_images = blend_images(output_images, parsed_arguments)
                
    return output_images

def average_frames(images):
    pass

def blend_images(images, parsed_arguments):
    
    perform_blend = False
    
    for key in parsed_arguments.keys():
        if "Blend_" in key:
            perform_blend = True
            break
    
    if perform_blend == False:
        return images
    
    output_images = [images[0]]
    
    for image_itr in range(len(images) - 1): # -1 because we don't need to blend anything to the final frame
        frame_1 = images[image_itr]
        frame_2 = images[image_itr + 1]
        blended_image = frame_2
        
        for blend_type in parsed_arguments.keys(): 
            # Doing a for loop here allows the blends to be ordered by the way they're ordered in the text box 
            # in case the end user wants multiple blends
            if blend_type == "Blend_Image.blend":
                blended_image = Image.blend(
                        frame_1, blended_image,
                        float(parsed_arguments["Blend_Image.blend"]) or 0.5)
            elif blend_type == "Blend_cv2.addWeighted":
                weight = float(parsed_arguments["Blend_cv2.addWeighted"]) or 0.5
                blended_image = Image.fromarray(cv2.addWeighted(np.asarray(frame_1), 1 - weight, np.asarray(blended_image), weight, 0), mode='RGB')
            elif blend_type == "Blend_interp1d":
                t = np.linspace(0, 1, 2)
                interpolator = interp1d(
                    t, [np.asarray(frame_1), np.asarray(blended_image)], axis=0, 
                    kind=parsed_arguments["interp1d_kind"] if "interp1d_kind" in parsed_arguments else "linear" or "linear", 
                    fill_value=parsed_arguments["interp1d_fill_value"] if "interp1d_fill_value" in parsed_arguments else "extrapolate" or "extrapolate") 
                blended_image = Image.fromarray(interpolator(float(parsed_arguments["Blend_interp1d"]) or 0.5).astype(np.uint8), mode='RGB')
            elif blend_type == "Blend_CubicSpline":
                t = np.linspace(0, 1, 2)
                spline = CubicSpline(t, [np.asarray(frame_1), np.asarray(blended_image)], axis=0)
                blended_image = Image.fromarray(spline(float(parsed_arguments["Blend_CubicSpline"]) or 0.5), mode='RGB')
            elif blend_type == "Blend_ImageChops.blend":
                blended_image = ImageChops.blend(
                        frame_1, blended_image,
                        float(parsed_arguments["Blend_ImageChops.blend"]))
            elif blend_type == "Blend_FrameAveraging":
                buffer_size = int(parsed_arguments["Blend_FrameAveraging"]) or 1
                images_left = (len(images) - 1) - image_itr
                buffer_size = min(buffer_size, image_itr, images_left) # get the least of these numbers
                frame_buffer = []
                for image_index in range(image_itr - buffer_size, image_itr + buffer_size):
                    frame_buffer.append(np.asarray(images[image_index]))
                if len(frame_buffer) > 2:
                    blended_image = Image.fromarray(np.mean(frame_buffer, axis=0).astype(np.uint8), mode='RGB')
                
            
        output_images.append(blended_image)
            
    return output_images

def get_remap_args_from_parsed_arguments(parsed_arguments):
        interp_mode = int(parsed_arguments["interp_mode"]) if "interp_mode" in parsed_arguments else cv2.INTER_LANCZOS4
        border_mode = int(parsed_arguments["border_mode"]) if "border_mode" in parsed_arguments else cv2.BORDER_REFLECT
        
        return interp_mode, border_mode
    
def generate_pil_frames_in_between(frame_1: Image.Image, frame_2: Image.Image, num_frames = 1, parsed_arguments = {}, *args) -> List[Image.Image]:
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
    
    previous_frame = np.asarray(frame_1)
    
    resultant_frames = []
    optical_flow = calculate_optical_flow_between_frames(previous_frame, np.asarray(frame_2), *args)
    optical_flow = median_filter(optical_flow)
    optical_flow = gaussian_smooth(optical_flow)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        
        interp_mode, border_mode = get_remap_args_from_parsed_arguments(parsed_arguments)
        interpolated_frame = cv2.remap(previous_frame, flow, None, interp_mode, borderMode=border_mode) # INTER_LANCZOS4 is p good
        
        resultant_frames.append(Image.fromarray(interpolated_frame, mode='RGB'))
    return resultant_frames

def generate_pil_frames_in_between_alt1(frame_1: Image.Image, frame_2: Image.Image, num_frames = 1, parsed_arguments = {}, *args) -> List[Image.Image]:
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
    
    previous_frame = np.asarray(frame_1)
    
    resultant_frames = []
    optical_flow = calculate_optical_flow_between_frames(previous_frame, np.asarray(frame_2), *args)
    optical_flow = median_filter(optical_flow)
    optical_flow = gaussian_smooth(optical_flow)
    h, w = optical_flow.shape[:2]
    
    previous_flow = None
    
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        
        if previous_flow is not None:
            optical_flow = alpha * optical_flow + (1 - alpha) * previous_flow
            previous_flow = optical_flow.copy()
            
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        
        interp_mode, border_mode = get_remap_args_from_parsed_arguments(parsed_arguments)
        interpolated_frame = cv2.remap(previous_frame, flow, None, interp_mode, borderMode=border_mode) # INTER_LANCZOS4 is p good
        
        resultant_frames.append(Image.fromarray(interpolated_frame, mode='RGB'))
    return resultant_frames

def generate_pil_frames_in_between_alt2(frame_1: Image.Image, frame_2: Image.Image, num_frames = 1, parsed_arguments = {}, *args) -> List[Image.Image]:
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
    
    previous_frame = np.asarray(frame_1)
    
    resultant_frames = []
    optical_flow = cv2.optflow.calcOpticalFlowDenseRLOF(previous_frame, np.asarray(frame_2), *args)
    optical_flow = median_filter(optical_flow)
    optical_flow = gaussian_smooth(optical_flow)
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames+1):
        alpha = frame_num / (num_frames + 1)
        flow =  -1 * alpha * optical_flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        
        interp_mode, border_mode = get_remap_args_from_parsed_arguments(parsed_arguments)
        interpolated_frame = cv2.remap(previous_frame, flow, None, interp_mode, borderMode=border_mode) # INTER_LANCZOS4 is p good
        
        resultant_frames.append(Image.fromarray(interpolated_frame, mode='RGB'))
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
        