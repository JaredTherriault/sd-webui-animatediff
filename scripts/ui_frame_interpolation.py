import os
import gradio as gr

def get_interpolation_methods():
    return ["Optical Flow"]

def get_interpolation_ui_controls():
    with gr.Accordion('Interpolation', open=False):
        with gr.Row():
            interpolation_enabled = gr.Checkbox(True, label="Interpolation Enabled", type="value")
            frame_multiplier = gr.Number(minimum=2, value=2, precision=0, label="Frame Multipler", type="value", info="This will multiply the frame rate by the specified amount, but will not affect the video length.")
        with gr.Row():
            custom_arguments = gr.Textbox(label="custom_arguments", type="text")
            
        with gr.Accordion('Advanced', open=False):
            method = gr.Dropdown(choices=get_interpolation_methods(), value=get_interpolation_methods()[0], label="Interpolation Method", type="value")
            with gr.Row():
                pyr_scale = gr.Number(minimum=0.0000001, maximum=0.9999999, value=0.5, step=0.01, label="pyr_scale", type="value", info="Scale of the image pyramid used in the optical flow calculation. Smaller value = finer pyramid = more levels = more detail but may be slower.")
                levels = gr.Number(minimum=1, value=3, precision=0, label="levels", type="value", info="Determines the number of levels in the image pyramid. More levels can capture motion at different scales but may increase computational cost.")
                winsize = gr.Number(minimum=1, value=16, precision=0, label="winsize", type="value", info="Defines the size of the pixel neighborhood used for flow integration. Larger winsize can handle larger motions but may blur small details. ")
            with gr.Row(): 
                iterations = gr.Number(minimum=1, value=3, precision=0, label="iterations", type="value", info="Controls the number of iterations at each pyramid level. More iterations can improve accuracy but increase computation time.")
                poly_n = gr.Number(minimum=1, value=5, precision=0, label="poly_n", type="value", info="Pixel neighborhood size used when estimating polynomial expansion. A higher value can capture more complex motion but may increase noise.")
                poly_sigma = gr.Number(minimum=0.0000001, value=1.5, step=0.1, label="poly_sigma", type="value", info="Standard deviation of Gaussian to smooth polynomial expansion derivatives. Larger value = stronger smoothing, which can reduce noise.")
                flags = gr.Number(minimum=0, value=0, precision=0, label="flags", type="value", info="Include additional flags to customize the optical flow calculation. The value of 0 typically means no additional flags are used.")
    return interpolation_enabled, frame_multiplier, custom_arguments, method, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags