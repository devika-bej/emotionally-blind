import cv2
import numpy as np
import os
import random
import math

# --- 1. Main Configuration ---

# Total number of images to generate
TOTAL_IMAGES = 20000 
# Dimensions for the square images
IMAGE_SIZE = (128, 128)  # Height, Width

# Base directory for the dataset
DATASET_DIR = 'dataset_fc'

# [cite_start]Define your simple elements [cite: 31]
FEARFUL_SHAPES = ['cross']
NEUTRAL_SHAPES = ['circle', 'triangle', 'line']

# Configure dataset splits
SPLITS = {
    'train': 0.7,
    'validation': 0.15,
    'test': 0.15
}

# [cite_start]Per[cite: 43], training set should be biased towards the fearful element
# We'll make 70% of training images 'fearful'
FEARFUL_BIAS_TRAIN = 0.7
# Validation and Test splits should be balanced for fair evaluation
FEARFUL_BIAS_EVAL = 0.5

# --- 2. Directory Creation Function ---

def create_directory_structure():
    """
    Creates the nested directory structure for train/validation/test splits
    and their 'fearful'/'neutral' subfolders.
    """
    print(f"Creating directory structure in '{DATASET_DIR}'...")
    for split in SPLITS.keys():
        for label in ['fearful', 'neutral']:
            path = os.path.join(DATASET_DIR, split, label)
            os.makedirs(path, exist_ok=True)
    print("Directory structure created successfully.")

# --- 3. Shape Drawing Functions ---

def get_random_shape_params():
    """Generates random parameters for drawing a shape."""
    # Center point, avoiding edges
    center_x = random.randint(IMAGE_SIZE[1] // 4, IMAGE_SIZE[1] * 3 // 4)
    center_y = random.randint(IMAGE_SIZE[0] // 4, IMAGE_SIZE[0] * 3 // 4)
    
    # Size of the shape
    radius = random.randint(IMAGE_SIZE[0] // 8, IMAGE_SIZE[0] // 3)
    
    # Color (white)
    color = 255 
    
    # [cite_start]"different thickness" [cite: 26]
    thickness = random.randint(2, 6)
    
    return (center_x, center_y), radius, color, thickness

def draw_circle(img, params):
    (center_x, center_y), radius, color, thickness = params
    cv2.circle(img, (center_x, center_y), radius, color, thickness)
    return img

def draw_line(img, params):
    (center_x, center_y), radius, color, thickness = params
    # Draw a line through the center point at a random angle
    angle = random.uniform(0, math.pi)
    pt1_x = int(center_x - radius * math.cos(angle))
    pt1_y = int(center_y - radius * math.sin(angle))
    pt2_x = int(center_x + radius * math.cos(angle))
    pt2_y = int(center_y + radius * math.sin(angle))
    cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, thickness)
    return img
        
def draw_triangle(img, params):
    (center_x, center_y), radius, color, thickness = params
    # Points for an equilateral triangle
    points = []
    for i in range(3):
        angle = (i * 2 * math.pi / 3) - (math.pi / 2) # Start pointing up
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        points.append((x, y))
    
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img

def draw_cross(img, params):
    (center_x, center_y), radius, color, thickness = params
    # Horizontal line
    cv2.line(img, (center_x - radius, center_y), (center_x + radius, center_y), color, thickness)
    # Vertical line
    cv2.line(img, (center_x, center_y - radius), (center_x, center_y + radius), color, thickness)
    return img

# Map shape names to their drawing functions
SHAPE_FUNCTIONS = {
    'circle': draw_circle,
    'triangle': draw_triangle,
    'line': draw_line,
    'cross': draw_cross
}

# [cite_start]--- 4. Augmentation Functions [cite: 24, 26] ---

def apply_blur(img):
    """Applies Gaussian blur with a random kernel size."""
    ksize = random.choice([(3,3), (5,5), (7,7)])
    return cv2.GaussianBlur(img, ksize, 0)

def apply_rotation(img):
    """Applies a random rotation between -45 and 45 degrees."""
    angle = random.randint(-45, 45)
    center = (IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use borderValue=0 to keep the background black
    return cv2.warpAffine(img, M, (IMAGE_SIZE[1], IMAGE_SIZE[0]), flags=cv2.INTER_LINEAR, borderValue=0)

def apply_occlusion(img):
    """Applies a random black rectangle to occlude part of the image."""
    occluded_img = img.copy()
    # [cite_start]"only part of the element is visible" [cite: 26]
    x1 = random.randint(0, IMAGE_SIZE[1] - 20)
    y1 = random.randint(0, IMAGE_SIZE[0] - 20)
    # Ensure a minimum occlusion size
    width = random.randint(20, IMAGE_SIZE[1] - x1)
    height = random.randint(20, IMAGE_SIZE[0] - y1)
    x2 = x1 + width
    y2 = y1 + height
    
    cv2.rectangle(occluded_img, (x1, y1), (x2, y2), 0, -1) # -1 thickness fills the rect
    return occluded_img

# --- 5. Main Dataset Generation Loop ---

def generate_dataset():
    """
    Main function to generate the complete dataset.
    """
    create_directory_structure()
    print(f"Generating {TOTAL_IMAGES} images...")
    
    # Keep track of counts for verification
    counts = {
        'train': {'fearful': 0, 'neutral': 0},
        'validation': {'fearful': 0, 'neutral': 0},
        'test': {'fearful': 0, 'neutral': 0}
    }
    
    # Pre-calculate split counts
    train_count = int(TOTAL_IMAGES * SPLITS['train'])
    val_count = int(TOTAL_IMAGES * SPLITS['validation'])
    
    img_counter = 0

    for i in range(TOTAL_IMAGES):
        # 1. Determine the split (train, val, or test)
        if i < train_count:
            split = 'train'
            fearful_bias = FEARFUL_BIAS_TRAIN
        elif i < train_count + val_count:
            split = 'validation'
            fearful_bias = FEARFUL_BIAS_EVAL
        else:
            split = 'test'
            fearful_bias = FEARFUL_BIAS_EVAL

        # 2. Determine the label (fearful or neutral)
        if random.random() < fearful_bias:
            label = 'fearful'
            shape_name = random.choice(FEARFUL_SHAPES)
        else:
            label = 'neutral'
            shape_name = random.choice(NEUTRAL_SHAPES)

        # 3. Create base image
        # Create a black background image (grayscale, 8-bit)
        img = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        # Get random parameters and draw the chosen shape
        params = get_random_shape_params()
        draw_func = SHAPE_FUNCTIONS[shape_name]
        img = draw_func(img, params)

        # [cite_start]4. Apply augmentations [cite: 24, 26]
        # We apply them randomly to create a varied dataset
        
        if random.random() < 0.4: # 40% chance of blur
            img = apply_blur(img)
            
        if random.random() < 0.6: # 60% chance of rotation
            img = apply_rotation(img)
            
        if random.random() < 0.3: # 30% chance of occlusion
            img = apply_occlusion(img)
            
        # 5. Save the final image
        img_counter += 1
        counts[split][label] += 1
        
        # Format filename with leading zeros
        filename = f"img_{img_counter:05d}.png" 
        filepath = os.path.join(DATASET_DIR, split, label, filename)
        
        cv2.imwrite(filepath, img)

        # Print progress
        if (i+1) % (TOTAL_IMAGES // 10) == 0:
            print(f"  ... generated {i+1} / {TOTAL_IMAGES} images")
    
    print("\n--- Dataset Generation Complete ---")
    print("Final image counts:")
    print(f"  Training:   {counts['train']}")
    print(f"  Validation: {counts['validation']}")
    print(f"  Test:       {counts['test']}")
    print(f"\nDataset saved to '{DATASET_DIR}'")

# --- 6. Run the Script ---

if __name__ == "__main__":
    generate_dataset()
