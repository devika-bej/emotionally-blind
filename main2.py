import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

# ==========================================
# 0. BATCH CONFIGURATION
# ==========================================
# Folder containing your subfolders of videos (e.g., /kaggle/input/fear-videos/)
INPUT_ROOT_DIR = "/kaggle/input/videos" 

# Where to save the processed videos
OUTPUT_ROOT_DIR = "/kaggle/working/processed_experiments"
MODEL_PATH = "/kaggle/working/fear_model.pth"

# ==========================================
# 1. Hebbian Model Definition
# ==========================================
class ImprovedHebbianFear(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, input_size))
        self.temperature = 50.0 

    def forward(self, masked_img_tensor):
        x_flat = masked_img_tensor.view(masked_img_tensor.size(0), -1)
        raw_activation = torch.mm(x_flat, self.weights.t()).item()
        scaled_input = (raw_activation / self.temperature) - 3.0
        probability = 1.0 / (1.0 + np.exp(-scaled_input))
        return probability

    def check_thresholds(self, score, t_detect=0.3, t_ident=0.7):
        if score >= t_ident: return "IDENTIFIED"
        elif score >= t_detect: return "DETECTED"
        else: return "NEUTRAL"

# ==========================================
# 2. Global State & Initialization
# ==========================================
IMG_SIZE = 64
pos_angles = [45, 135, 225, 315]
pos_dists = [200, 400, 600]

# These will be reset for every video
current_verdict = "NEUTRAL"
neutral_detected = False
detected_frame_countdown = -1
identified_frame_countdown = -1

# Load Model
model = ImprovedHebbianFear(IMG_SIZE * IMG_SIZE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded weights from {MODEL_PATH}")
else:
    print("WARNING: Using random weights (Simulation Mode)")
    nn.init.normal_(model.weights, mean=0.5, std=0.1)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x) 
])

# ==========================================
# 3. Helper Functions
# ==========================================
def reset_state():
    """Resets the 'brain' state for a new video."""
    global current_verdict, neutral_detected, detected_frame_countdown, identified_frame_countdown
    current_verdict = "NEUTRAL"
    neutral_detected = False
    detected_frame_countdown = -1
    identified_frame_countdown = -1

def get_roi_patch(frame, center, size=64):
    x, y = int(center[0]), int(center[1])
    r = size // 2
    h, w, _ = frame.shape
    padded = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT, value=255)
    x_pad, y_pad = x + r, y + r
    patch = padded[x_pad - r : x_pad + r, y_pad - r : y_pad + r]
    return patch

def update_verdict(frame, focus_center):
    global current_verdict, model
    patch = get_roi_patch(frame, focus_center, size=IMG_SIZE)
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray_patch)
    input_tensor = preprocess(pil_img).unsqueeze(0) 
    
    fear_score = model(input_tensor)
    current_verdict = model.check_thresholds(fear_score)
    
    return fear_score # Return score for stats logging

def create_2d_gaussian_mask(image_shape, center, sigma_x, sigma_y, angle=0):
    h, w = image_shape
    y, x = np.ogrid[0:h, 0:w]
    y0, x0 = center
    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = (x - x0) * cos_t + (y - y0) * sin_t
    y_rot = -(x - x0) * sin_t + (y - y0) * cos_t
    mask = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    return mask

# ==========================================
# 4. Frame Processing Logic
# ==========================================
def mask_frame(frame):
    global current_verdict, neutral_detected, detected_frame_countdown, identified_frame_countdown
    
    center = (frame.shape[0] // 2, frame.shape[1] // 2)
    gauss_center = center
    gauss_angle = 0
    gauss_dist_x = 50
    gauss_dist_y = 50
    got_gauss = False
    max_fear_score = 0.0

    # Peripheral Scan
    for a in pos_angles:
        if got_gauss: break
        for d in pos_dists:
            a_rad = np.radians(a)
            x = center[0] + int(d * np.cos(a_rad))
            y = center[1] + int(d * np.sin(a_rad))
            if x < 10 or x > frame.shape[0]-10 or y < 10 or y > frame.shape[1]-10: continue
            patch = frame[x - 10 : x + 10, y - 10 : y + 10]
            if np.any(patch < 50): 
                got_gauss = True
                gauss_center = ((center[0] + x) / 2, (center[1] + y) / 2) 
                gauss_angle = a
                gauss_dist_x = np.abs(x - gauss_center[0]) + 50 
                gauss_dist_y = np.abs(y - gauss_center[1]) + 50
                break

    # State Machine
    if current_verdict == "NEUTRAL":
        if got_gauss and not neutral_detected:
            max_fear_score = update_verdict(frame, gauss_center) # Capture score
            if current_verdict == "NEUTRAL":
                neutral_detected = True
            elif current_verdict == "DETECTED":
                detected_frame_countdown = 5
        target_center = gauss_center if (got_gauss and not neutral_detected) else center
        sigma_x = gauss_dist_x if (got_gauss and not neutral_detected) else 50
        sigma_y = gauss_dist_y if (got_gauss and not neutral_detected) else 50
        mask = create_2d_gaussian_mask(frame.shape[:2], target_center, sigma_x, sigma_y, gauss_angle)

    elif current_verdict == "DETECTED":
        if detected_frame_countdown < 0:
            max_fear_score = update_verdict(frame, center) # Capture score
            if current_verdict != "IDENTIFIED":
                current_verdict = "NEUTRAL"
        else:
            detected_frame_countdown -= 1
        mask = create_2d_gaussian_mask(frame.shape[:2], gauss_center, gauss_dist_x, gauss_dist_y, gauss_angle)

    elif current_verdict == "IDENTIFIED":
        if got_gauss:
            identified_frame_countdown = 10
            max_fear_score = update_verdict(frame, gauss_center) # Capture score
        else:
            if identified_frame_countdown < 0:
                current_verdict = "NEUTRAL"
            else:
                identified_frame_countdown -= 1
        mask = create_2d_gaussian_mask(frame.shape[:2], gauss_center, gauss_dist_x, gauss_dist_y, gauss_angle)

    # Render
    masked_img = frame.copy()
    for i in range(3):
        masked_img[:, :, i] = frame[:, :, i] * mask

    return masked_img, max_fear_score

# ==========================================
# 5. Batch Execution Engine
# ==========================================
def process_single_video(video_path, save_path):
    reset_state() # CRITICAL: Wipe memory of previous video
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Safety check for empty/corrupt videos
    if width == 0 or height == 0: return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    stats = {"fear_frames": 0, "total_frames": 0, "max_score": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        processed_frame, score = mask_frame(frame)
        out.write(processed_frame)
        
        stats["total_frames"] += 1
        stats["max_score"] = max(stats["max_score"], score)
        if current_verdict == "IDENTIFIED":
            stats["fear_frames"] += 1

    cap.release()
    out.release()
    return stats

def main_batch():
    if not os.path.exists(INPUT_ROOT_DIR):
        print(f"Error: Input directory {INPUT_ROOT_DIR} does not exist.")
        return

    # Find all mp4 files recursively
    video_files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**/*.mp4"), recursive=True)
    print(f"Found {len(video_files)} videos to process.")
    
    report_log = []

    for i, v_path in enumerate(video_files):
        # Create corresponding output path
        rel_path = os.path.relpath(v_path, INPUT_ROOT_DIR)
        out_path = os.path.join(OUTPUT_ROOT_DIR, rel_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        print(f"[{i+1}/{len(video_files)}] Processing: {rel_path}...")
        
        stats = process_single_video(v_path, out_path)
        
        if stats:
            fear_percentage = (stats['fear_frames'] / stats['total_frames']) * 100 if stats['total_frames'] > 0 else 0
            log_entry = f"{rel_path:<40} | Max Fear: {stats['max_score']:.2f} | Panic Time: {fear_percentage:.1f}%"
            print(f"   -> Done. {log_entry}")
            report_log.append(log_entry)
        else:
            print("   -> Failed to process.")

    print("\n" + "="*60)
    print("FINAL BATCH REPORT")
    print("="*60)
    for line in report_log:
        print(line)

if __name__ == "__main__":
    main_batch()
