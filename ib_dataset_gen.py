import cv2
import numpy as np
import os
import random
import glob, os
import re

# --- 1. Configuration Parameters ----------------------------------------------
# You can adjust these settings

# Video Specs
WIDTH = 256
HEIGHT = 256
FPS = 30
DURATION_SEC = 15
TOTAL_FRAMES = FPS * DURATION_SEC

# Output Directories
INPUT_DIR = "input_frames/"
GT_DIR = "gt_frames/"
VID_DIR = "videos/"

# Ground Truth Probabilities
RECALL_NEUTRAL = 0.25  # Prob. to mask triangle when present
RECALL_FEARFUL = 0.35  # Prob. to mask cross when present

# Object Appearance Probabilities (per-frame)
TRIANGLE_APPEAR_PROB = 0.05  # Chance to appear if not visible
CROSS_APPEAR_PROB = 0.015   # Chance to appear if not visible

# Object Physics & Style
CIRCLE_RADIUS = 10
CIRCLE_MAX_SPEED = 4
TRIANGLE_SIZE = 15  # "Radius" from center to vertex
CROSS_SIZE = 12     # Half-length of a cross arm
CROSS_THICKNESS = 3
TRIANGLE_THICKNESS = -1 # -1 fills the shape

# --- 2. Helper Functions for Drawing -----------------------------------------

def draw_triangle(frame, center, size, color, thickness):
    """Draws a filled upright triangle centered at 'center'."""
    cx, cy = center
    # Vertices of the triangle
    points = np.array([
        [cx, cy - size],             # Top point
        [cx - size, cy + size],      # Bottom-left
        [cx + size, cy + size]       # Bottom-right
    ], dtype=np.int32)
    
    cv2.fillPoly(frame, [points], color)

def draw_cross(frame, center, size, color, thickness):
    """Draws a '+' cross centered at 'center'."""
    cx, cy = center
    # Horizontal line
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    # Vertical line
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

# --- 3. Main State and Generation Logic --------------------------------------

def generate_dataset(index):
    print(f"Generating dataset in '{INPUT_DIR}/' and '{GT_DIR}/'...")
    
    # Create output directories
    input_dir = f"{INPUT_DIR}{index}/"
    gt_dir = f"{GT_DIR}{index}/"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # --- Initialize Object States ---
    
    # Circle state
    circle_pos = [WIDTH // 2, HEIGHT // 2]
    circle_vel = [random.randint(-CIRCLE_MAX_SPEED, CIRCLE_MAX_SPEED), 
                  random.randint(-CIRCLE_MAX_SPEED, CIRCLE_MAX_SPEED)]
    # Ensure it's not stationary
    if circle_vel[0] == 0 and circle_vel[1] == 0:
        circle_vel[0] = 1

    # Triangle state
    triangle_visible = False
    triangle_frames_left = 0
    triangle_pos = [0, 0]
    mask_triangle = -1
    mask_triangle_timer = FPS

    # Cross state
    cross_visible = False
    cross_frames_left = 0
    cross_pos = [0, 0]
    mask_cross = -1
    mask_cross_timer = 2 * FPS
    
    # --- Main Generation Loop ---
    for i in range(TOTAL_FRAMES):
        
        # Create black canvases for input and ground truth
        input_frame = np.full((HEIGHT, WIDTH, 3), (0, 255, 0), dtype=np.uint8)
        gt_mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Reset per-frame flags
        circle_is_bouncing = False
        
        # --- 3.1. Update and Draw Circle ---
        
        # Update position
        circle_pos[0] += circle_vel[0]
        circle_pos[1] += circle_vel[1]

        # Wall bounce logic
        if circle_pos[0] - CIRCLE_RADIUS <= 0 or \
           circle_pos[0] + CIRCLE_RADIUS >= WIDTH:
            circle_vel[0] = -circle_vel[0]
            circle_pos[0] = np.clip(circle_pos[0], CIRCLE_RADIUS, WIDTH - CIRCLE_RADIUS)
            circle_is_bouncing = True
            
        if circle_pos[1] - CIRCLE_RADIUS <= 0 or \
           circle_pos[1] + CIRCLE_RADIUS >= HEIGHT:
            circle_vel[1] = -circle_vel[1]
            circle_pos[1] = np.clip(circle_pos[1], CIRCLE_RADIUS, HEIGHT - CIRCLE_RADIUS)
            circle_is_bouncing = True

        # Draw circle on input frame
        pos_int = (int(circle_pos[0]), int(circle_pos[1]))
        cv2.circle(input_frame, pos_int, CIRCLE_RADIUS, (255, 255, 255), -1)

        # **Update GT Mask (Circle)**
        if True:
            x_min_c = max(0, pos_int[0] - CIRCLE_RADIUS)
            x_max_c = min(WIDTH, pos_int[0] + CIRCLE_RADIUS)
            y_min_c = max(0, pos_int[1] - CIRCLE_RADIUS)
            y_max_c = min(HEIGHT, pos_int[1] + CIRCLE_RADIUS)
            # Copy pixels from input_frame to gt_mask
            gt_mask[y_min_c:y_max_c, x_min_c:x_max_c] = input_frame[y_min_c:y_max_c, x_min_c:x_max_c]
            
        # --- 3.2. Update and Draw Triangle ---
        
        if triangle_visible:
            # Draw it
            draw_triangle(input_frame, triangle_pos, TRIANGLE_SIZE, 
                          (255, 255, 255), TRIANGLE_THICKNESS)
            
            # **Update GT Mask (Triangle)**
            if mask_triangle < 0:
                mask_triangle = random.random()
            
            if mask_triangle < RECALL_NEUTRAL and mask_triangle_timer > 0:
                # Define triangle bounding box
                tx, ty = int(triangle_pos[0]), int(triangle_pos[1])
                x_min_t = max(0, tx - TRIANGLE_SIZE)
                x_max_t = min(WIDTH, tx + TRIANGLE_SIZE)
                y_min_t = max(0, ty - TRIANGLE_SIZE)
                y_max_t = min(HEIGHT, ty + TRIANGLE_SIZE)
                # Copy pixels from input_frame to gt_mask
                gt_mask[y_min_t:y_max_t, x_min_t:x_max_t] = input_frame[y_min_t:y_max_t, x_min_t:x_max_t]
                mask_triangle_timer -= 1
                if mask_triangle_timer <= 0:
                    mask_triangle = -1
                    mask_triangle_timer = FPS
            
            # Update timer
            triangle_frames_left -= 1
            if triangle_frames_left <= 0:
                triangle_visible = False
        else:
            # Check if it should appear
            if random.random() < TRIANGLE_APPEAR_PROB:
                triangle_visible = True
                triangle_frames_left = random.randint(4 * FPS, 6 * FPS)
                triangle_pos = [random.randint(TRIANGLE_SIZE, WIDTH - TRIANGLE_SIZE), 
                                random.randint(TRIANGLE_SIZE, HEIGHT - TRIANGLE_SIZE)]

        # --- 3.3. Update and Draw Cross ---

        if cross_visible:
            # Draw it
            draw_cross(input_frame, cross_pos, CROSS_SIZE, 
                       (255, 255, 255), CROSS_THICKNESS)
            
            # **Update GT Mask (Cross)**
            if mask_cross < 0:
                mask_cross = random.random()
            
            if mask_cross < RECALL_FEARFUL and mask_cross_timer > 0:
                # Define cross bounding box
                cx, cy = int(cross_pos[0]), int(cross_pos[1])
                x_min_x = max(0, cx - CROSS_SIZE)
                x_max_x = min(WIDTH, cx + CROSS_SIZE)
                y_min_x = max(0, cy - CROSS_SIZE)
                y_max_x = min(HEIGHT, cy + CROSS_SIZE)
                # Copy pixels from input_frame to gt_mask
                gt_mask[y_min_x:y_max_x, x_min_x:x_max_x] = input_frame[y_min_x:y_max_x, x_min_x:x_max_x]
                mask_cross_timer -= 1
                if mask_cross_timer <= 0:
                    mask_cross = -1
                    mask_cross_timer = 2 * FPS
            
            # Update timer
            cross_frames_left -= 1
            if cross_frames_left <= 0:
                cross_visible = False
        else:
            # Check if it should appear
            if random.random() < CROSS_APPEAR_PROB:
                cross_visible = True
                cross_frames_left = random.randint(2 * FPS, 4 * FPS)
                cross_pos = [random.randint(CROSS_SIZE, WIDTH - CROSS_SIZE), 
                             random.randint(CROSS_SIZE, HEIGHT - CROSS_SIZE)]

        # --- 3.4. Save Frames ---
        
        input_filename = os.path.join(input_dir, f"frame_{i:04d}.png")
        gt_filename = os.path.join(gt_dir, f"mask_{i:04d}.png")
        
        cv2.imwrite(input_filename, input_frame)
        cv2.imwrite(gt_filename, gt_mask)
        
        if (i + 1) % FPS == 0:
            print(f"Generated frame {i + 1} / {TOTAL_FRAMES}")

    print("\nDataset generation complete.")
    print(f"Input frames: {os.path.abspath(input_dir)}")
    print(f"GT masks:     {os.path.abspath(gt_dir)}")

def compile_video(frame_dir, output_path, fps, width, height):
    """
    Compiles a video from a directory of image frames.
    """
    # Get all .png frame files, sorted numerically
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    
    if not frame_files:
        print(f"Error: No .png frames found in {frame_dir}")
        return

    print(f"Compiling {len(frame_files)} frames from '{frame_dir}' into '{output_path}'...")
    
    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path {output_path}")
        return

    for frame_path in frame_files:
        img = cv2.imread(frame_path)
        
        # Check if image was read correctly
        if img is None:
            print(f"Warning: Skipping missing or corrupted frame {frame_path}")
            continue
            
        # Ensure frame is the correct size (just in case)
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
            
        video_writer.write(img)

    video_writer.release()
    print(f"Successfully saved video: {os.path.abspath(output_path)}")


def save_generated_videos(index, fps, width, height):
    """
    Saves both the input frames and GT frames as separate videos.
    
    Args:
        input_dir (str): Path to the 'input_frames' directory.
        gt_dir (str): Path to the 'gt_frames' directory.
        output_prefix (str): The path and base name for the videos
                             (e.g., '../my_video' or 'videos/dataset1')
        fps (int): Frames per second for the output video.
        width (int): Frame width.
        height (int): Frame height.
    """
    
    input_dir = f"{INPUT_DIR}{index}/"
    gt_dir = f"{GT_DIR}{index}/"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(VID_DIR)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compile the input video
    input_video_path = f"{VID_DIR}{index}_input.mp4"
    compile_video(input_dir, input_video_path, fps, width, height)
    
    # Compile the ground truth video
    gt_video_path = f"{VID_DIR}{index}_gt.mp4"
    compile_video(gt_dir, gt_video_path, fps, width, height)

# --- 4. Run the Generator ----------------------------------------------------

if __name__ == "__main__":
    num_vids = 10
    for index in range(num_vids):
        generate_dataset(index)
        save_generated_videos(index, FPS, WIDTH, HEIGHT)