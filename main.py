import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

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
        if score >= t_ident:
            return "IDENTIFIED"  # High Fear (Danger)
        elif score >= t_detect:
            return "DETECTED"    # Suspicion (Warning)
        else:
            return "NEUTRAL"     # Safe

# ==========================================
# 2. Global State & Config
# ==========================================
IMG_SIZE = 64
pos_angles = [45, 135, 225, 315]
pos_dists = [200, 400, 600]

current_verdict = "NEUTRAL"
neutral_detected = False
detected_frame_countdown = -1
identified_frame_countdown = -1

# Initialize Model
model = ImprovedHebbianFear(IMG_SIZE * IMG_SIZE)

# Attempt to load trained weights
try:
    # If you saved your model in the previous script as 'fear_model.pth'
    model.load_state_dict(torch.load("fear_model.pth"))
    print("Loaded trained fear weights.")
except FileNotFoundError:
    print("WARNING: No trained model found. Using random weights (Running in simulation mode).")
    # Initialize with random noise just so it runs
    nn.init.normal_(model.weights, mean=0.5, std=0.1)

# Preprocessing transform (Invert colors: Ink=1, BG=0)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x) 
])

# ==========================================
# 3. Helper Functions
# ==========================================
def get_roi_patch(frame, center, size=64):
    """
    Extracts a square patch from the frame centered at 'center'.
    This simulates the fovea focusing on a specific object.
    """
    x, y = int(center[0]), int(center[1])
    r = size // 2
    
    # Pad frame to handle edge cases
    h, w, _ = frame.shape
    padded = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT, value=255)
    
    # Adjust coordinates for padding
    x_pad, y_pad = x + r, y + r
    patch = padded[x_pad - r : x_pad + r, y_pad - r : y_pad + r]
    return patch

def update_verdict(frame, focus_center):
    """
    Extracts the object at the focus point and asks the Hebbian model
    if it is dangerous.
    """
    global current_verdict, model
    
    # 1. Get the visual patch (What is the eye looking at?)
    patch = get_roi_patch(frame, focus_center, size=IMG_SIZE)
    
    # 2. Preprocess for PyTorch (BGR -> Gray -> PIL -> Tensor)
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray_patch)
    input_tensor = preprocess(pil_img).unsqueeze(0) # Add batch dim

    # 3. Neural Inference
    fear_score = model(input_tensor)
    new_verdict = model.check_thresholds(fear_score)
    
    # Only update global state if the new verdict is "worse" or if we are refreshing
    # (Simplified logic: just take the model's opinion)
    current_verdict = new_verdict
    
    print(f"Foveating at {focus_center}: Fear Score {fear_score:.3f} -> {current_verdict}")

def create_2d_gaussian_mask(image_shape, center, sigma_x, sigma_y, angle=0):
    h, w = image_shape
    y, x = np.ogrid[0:h, 0:w]
    y0, x0 = center
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = (x - x0) * cos_theta + (y - y0) * sin_theta
    y_rot = -(x - x0) * sin_theta + (y - y0) * cos_theta
    mask = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    return mask

# ==========================================
# 4. Main Processing Logic
# ==========================================
def mask_frame(frame):
    center = (frame.shape[0] // 2, frame.shape[1] // 2)
    gauss_center = center
    gauss_angle = 0
    gauss_dist_x = 50
    gauss_dist_y = 50
    got_gauss = False

    # --- PERIPHERAL SCANNING ---
    for a in pos_angles:
        if got_gauss: break
        for d in pos_dists:
            a_rad = np.radians(a)
            x = center[0] + int(d * np.cos(a_rad))
            y = center[1] + int(d * np.sin(a_rad))
            
            # Boundary check
            if x < 10 or x > frame.shape[0]-10 or y < 10 or y > frame.shape[1]-10:
                continue

            # Check for black pixels in periphery
            patch = frame[x - 10 : x + 10, y - 10 : y + 10]
            # Simple threshold: is there any dark ink?
            if np.any(patch < 50): 
                got_gauss = True
                gauss_center = ((center[0] + x) / 2, (center[1] + y) / 2) # Saccade towards it
                gauss_angle = a
                gauss_dist_x = np.abs(x - gauss_center[0]) + 50 # Expand focus
                gauss_dist_y = np.abs(y - gauss_center[1]) + 50
                break

    global current_verdict, neutral_detected, detected_frame_countdown, identified_frame_countdown

    # --- STATE MACHINE ---
    
    # STATE: NEUTRAL (Searching)
    if current_verdict == "NEUTRAL":
        if got_gauss and not neutral_detected:
            # We saw something! Check what it is using the Model.
            update_verdict(frame, gauss_center)
            
            # If the model says it's neutral, ignore it for a while
            if current_verdict == "NEUTRAL":
                neutral_detected = True
            elif current_verdict == "DETECTED":
                detected_frame_countdown = 5
        
        # Apply Mask
        target_center = gauss_center if (got_gauss and not neutral_detected) else center
        sigma_x = gauss_dist_x if (got_gauss and not neutral_detected) else 50
        sigma_y = gauss_dist_y if (got_gauss and not neutral_detected) else 50
        
        mask = create_2d_gaussian_mask(frame.shape[:2], target_center, sigma_x, sigma_y, gauss_angle)

    # STATE: DETECTED (Suspicious - Double Check)
    elif current_verdict == "DETECTED":
        if detected_frame_countdown < 0:
            # Timer ran out, double check the object
            update_verdict(frame, center) # Assuming center has moved to object
            if current_verdict != "IDENTIFIED":
                current_verdict = "NEUTRAL"
        else:
            detected_frame_countdown -= 1
            
        mask = create_2d_gaussian_mask(frame.shape[:2], gauss_center, gauss_dist_x, gauss_dist_y, gauss_angle)

    # STATE: IDENTIFIED (Fear Response - Lock On)
    elif current_verdict == "IDENTIFIED":
        if got_gauss:
            identified_frame_countdown = 10
            # Keep verifying periodically
            update_verdict(frame, gauss_center)
        else:
            if identified_frame_countdown < 0:
                current_verdict = "NEUTRAL"
            else:
                identified_frame_countdown -= 1
        
        mask = create_2d_gaussian_mask(frame.shape[:2], gauss_center, gauss_dist_x, gauss_dist_y, gauss_angle)

    # Apply the calculated mask to the frame
    masked_img = frame.copy()
    for i in range(3):
        masked_img[:, :, i] = frame[:, :, i] * mask

    return masked_img

def main(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing {input_video_path}...")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        masked_frame = mask_frame(frame)
        out.write(masked_frame)

        i += 1
        if i % 30 == 0:
            print(f"Frame {i}: Current State = {current_verdict}")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video.mp4>")
    else:
        main(sys.argv[1], "output.mp4")
