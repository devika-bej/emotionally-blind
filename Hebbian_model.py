import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ==========================================
# 1. Configuration
# ==========================================
DATASET_DIR = "/kaggle/working/synthetic_fear_dataset"
IMG_SIZE = 64
LEARNING_RATE = 0.05
EPOCHS = 8

# Fear Hierarchy
CONDITIONING_STRENGTHS = {
    'spider': 1.0,
    'needle': 0.8,
    'scrambled_spider': 0.6,
    'scrambled_needle': 0.4,
    'rectilinear_spider': 0.2
}

# ==========================================
# 2. Dataset Loader
# ==========================================
class FearDataset(Dataset):
    def __init__(self, root_dir, target_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root_dir):
            print(f"Warning: {root_dir} not found.")
        else:
            for category in os.listdir(root_dir):
                cat_path = os.path.join(root_dir, category)
                if os.path.isdir(cat_path):
                    if category in target_map:
                        target = target_map[category]
                        for img_name in os.listdir(cat_path):
                            if img_name.lower().endswith(('.png', '.jpg')):
                                self.samples.append({
                                    'path': os.path.join(cat_path, img_name),
                                    'label': category,
                                    'target': target
                                })
            print(f"Loaded {len(self.samples)} real samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['path']).convert('L')
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(item['target'], dtype=torch.float32)
        return img, item['label'], target

# Transform: Invert (Ink=1, BG=0)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x) 
])

# ==========================================
# 3. Improved Hebbian Model
# ==========================================
class ImprovedHebbianFear(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, input_size))
        self.temperature = 50.0 

    def train_step(self, img_tensor, conditioning_strength, eta=0.05, decay=0.001):
        with torch.no_grad():
            x_flat = img_tensor.view(img_tensor.size(0), -1)
            hebbian_term = x_flat * conditioning_strength.unsqueeze(1)
            decay_term = decay * self.weights
            delta_w = eta * (hebbian_term - decay_term)
            self.weights += torch.mean(delta_w, dim=0, keepdim=True)

    def forward(self, masked_img_tensor):
        x_flat = masked_img_tensor.view(masked_img_tensor.size(0), -1)
        # Raw Energy
        raw_activation = torch.mm(x_flat, self.weights.t()).item()
        # Sigmoid Squash
        scaled_input = (raw_activation / self.temperature) - 3.0
        probability = 1.0 / (1.0 + np.exp(-scaled_input))
        return probability

    def check_thresholds(self, score, t_detect=0.3, t_ident=0.7):
        if score >= t_ident:
            return "DANGER"
        elif score >= t_detect:
            return "WARNING"
        else:
            return "Safe"

# ==========================================
# 4. Helpers
# ==========================================
def apply_fading_mask(img_tensor, intensity=1.0):
    return img_tensor * intensity

# ==========================================
# 5. Experiment Loop
# ==========================================
def run_improved_experiment():
    full_dataset = FearDataset(DATASET_DIR, CONDITIONING_STRENGTHS, transform)
    
    if len(full_dataset) == 0:
        print("Dataset missing. Please generate it first.")
        return

    dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    model = ImprovedHebbianFear(IMG_SIZE * IMG_SIZE)
    
    print("\n--- 1. TRAINING PHASE ---")
    for epoch in range(EPOCHS):
        for imgs, _, targets in dataloader:
            model.train_step(imgs, targets, eta=LEARNING_RATE)
            
    print("Training Complete.")
    
    # ---------------------------------------------------------
    # NEW: Specific Mask Intensity Test
    # ---------------------------------------------------------
    
    # SETTINGS: Change this value to simulate position!
    # 1.0 = Center (Brightest)
    # 0.5 = Mid-way (Faded)
    # 0.2 = Edge (Very Dark)
    TEST_MASK_INTENSITY = 1.0
    
    print(f"\n--- 2. VIDEO SIMULATION AT FIXED POSITION ---")
    print(f"Mask Intensity: {TEST_MASK_INTENSITY} (Simulating object slightly off-center)")
    print(f"{'Object Type':<20} | {'Raw Prob (0-1)':<15} | {'Action Triggered'}")
    print("-" * 60)
    
    # We collect one representative sample for each category
    tested_cats = []
    
    # Iterate through dataset to find one of each type
    for img, label, _ in full_dataset:
        if label not in tested_cats:
            
            # 1. Apply the Fading Mask to this specific object
            masked_input = apply_fading_mask(img.unsqueeze(0), TEST_MASK_INTENSITY)
            
            # 2. Get Score
            score = model.forward(masked_input)
            action = model.check_thresholds(score)
            
            print(f"{label:<20} | {score:.4f}          | {action}")
            tested_cats.append(label)
            
            # Stop once we have tested all categories defined in hierarchy
            if len(tested_cats) >= len(CONDITIONING_STRENGTHS):
                break

    # Visual Verification
    plt.figure(figsize=(5,5))
    w_img = model.weights.view(IMG_SIZE, IMG_SIZE).detach().numpy()
    plt.imshow(w_img, cmap='hot')
    plt.title("Learned Fear Memory")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_improved_experiment()
