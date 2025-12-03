import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# 1. Configuration
# ==========================================
DATASET_DIR = "synthetic_fear_dataset"
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
LEARNING_RATE = 0.05
EPOCHS = 5 

# Fear Hierarchy
CONDITIONING_STRENGTHS = {
    'spider': 1.0,
    'needle': 0.8,
    'scrambled_spider': 0.6,
    'scrambled_needle': 0.4,
    'rectilinear_spider': 0.2,
    'neutral': 0.0 
}

# ==========================================
# 2. Dataset Loader
# ==========================================
class FearDataset(Dataset):
    def __init__(self, root_dir, target_map, transform=None):
        self.samples = []
        self.transform = transform
            
        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            if os.path.isdir(cat_path):
                # Default to 0.0 if category not in map
                target = target_map.get(category, 0.0) 
                
                for img_name in os.listdir(cat_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(cat_path, img_name),
                            'label': category,
                            'target': target
                        })
        print(f"Found {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['path']).convert('L')
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(item['target'], dtype=torch.float32)
        return img, target

# ==========================================
# 3. Model Definition
# ==========================================
class ImprovedHebbianFear(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, input_size))
        
    def train_step(self, img_tensor, conditioning_strength, eta=0.05, decay=0.001):
        """
        Hebbian Update Rule: Delta_W = Learning_Rate * (Input * Output_Signal - Decay * Current_Weights)
        """
        with torch.no_grad():
            x_flat = img_tensor.view(img_tensor.size(0), -1)
            hebbian_term = x_flat * conditioning_strength.unsqueeze(1)
            
            decay_term = decay * self.weights
            
            delta_w = eta * (hebbian_term - decay_term)
            self.weights += torch.mean(delta_w, dim=0, keepdim=True)

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])

    try:
        dataset = FearDataset(DATASET_DIR, CONDITIONING_STRENGTHS, transform)
    except FileNotFoundError as e:
        print(e)
        return

    if len(dataset) == 0:
        print("No images found.")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) 

    input_size = IMG_WIDTH * IMG_HEIGHT
    print(f"Initializing model with {input_size} synapses...")
    model = ImprovedHebbianFear(input_size)

    # Train
    for epoch in range(EPOCHS):
        total_strength = 0
        for imgs, targets in dataloader:
            model.train_step(imgs, targets, eta=LEARNING_RATE)
            total_strength += targets.sum().item()
        print(f"Epoch {epoch+1}/{EPOCHS}")

    # Save
    save_path = "fear_model_1080p.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    train()
