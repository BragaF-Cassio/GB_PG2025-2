import cv2
import numpy as np
import dearpygui.dearpygui as dpg

# DPG texture dimensions (can be changed to match video source or desired display size)
# A 640x480 resolution is a common default for webcams
texture_width = 640
texture_height = 480

# 1. Initialize OpenCV Video Capture (0 for webcam, or "video.mp4" for file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

# Set the capture dimensions for consistency if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, texture_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, texture_height)


# 2. Setup Dear PyGui
dpg.create_context()

# Create a texture registry and add a dynamic texture placeholder
with dpg.texture_registry(show=False):
    # We create a 1D float array for the dynamic texture
    # The size is width * height * 4 (for RGBA, although we use RGB data)
    # DPG expects floats in the [0.0, 1.0] range
    initial_data = np.zeros((texture_width * texture_height * 3), dtype=np.float32)
    dpg.add_dynamic_texture(texture_width, texture_height, initial_data, tag="video_texture")

# Create the main window and add an image item that uses the texture tag
with dpg.window(label="OpenCV Video Display", width=texture_width + 20, height=texture_height + 40):
    dpg.add_image("video_texture")

dpg.create_viewport(title='Custom Video Player', width=texture_width + 50, height=texture_height + 80)
dpg.setup_dearpygui()
dpg.show_viewport()


# 3. Define a custom render loop to update the frame
def update_video_frame():
    ret, frame = cap.read()
    if ret:
        # Convert BGR frame to RGB (DPG expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        frame_normalized = frame_rgb.flatten() / 255.0

        # Update the dynamic texture with the new frame data
        dpg.set_value("video_texture", frame_normalized)


# 4. Start the DPG main loop and clean up resources when finished
# Note: This is different from the standard dpg.start_dearpygui() because we control the loop
try:
    while dpg.is_dearpygui_running():
        update_video_frame()
        dpg.render_dearpygui_frame() 
except SystemExit:
    # Handle the application closing (e.g., hitting the 'X' button)
    pass
finally:
    # Release the OpenCV capture and destroy context
    cap.release()
    dpg.destroy_context()
