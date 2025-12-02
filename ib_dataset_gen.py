import cv2
import numpy as np
import os
import random
from PIL import Image

# Configuration
VIDEO_DURATION = 60  # seconds
FPS = 30
WIDTH, HEIGHT = 1920, 1080
BACKGROUND_COLOR = (255, 255, 255)  # White
SYMBOL_DURATION = 1  # seconds
NUM_SYMBOLS = 30
SYMBOL_SIZE = 100  # Base size for symbols

# Position configuration
ANGLES = [45, 135, 225, 315]  # degrees
DISTANCES = [200, 400, 600]  # pixels from center

def load_symbols(folder_path):
    """Load all symbol images from the specified folder."""
    symbols = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGBA')
            symbols.append(img)
    return symbols

def calculate_position(angle, distance, center):
    """Calculate x, y position based on angle and distance from center."""
    angle_rad = np.radians(angle)
    x = center[0] + int(distance * np.cos(angle_rad))
    y = center[1] + int(distance * np.sin(angle_rad))
    return (x, y)

def resize_and_rotate_symbol(symbol, size, rotation=0):
    """Resize and optionally rotate a symbol."""
    # Resize maintaining aspect ratio
    symbol.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    # Random rotation
    if rotation != 0:
        symbol = symbol.rotate(rotation, expand=True)
    
    return symbol

def overlay_symbol(frame, symbol, position):
    """Overlay a symbol (PIL Image) onto a frame (numpy array) at given position."""
    symbol_np = np.array(symbol)
    
    # Get symbol dimensions
    h, w = symbol_np.shape[:2]
    
    # Calculate top-left corner (position is center of symbol)
    x = position[0] - w // 2
    y = position[1] - h // 2
    
    # Ensure symbol is within frame bounds
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    
    # Handle alpha channel for transparency
    if symbol_np.shape[2] == 4:
        alpha = symbol_np[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                alpha * symbol_np[:, :, c] + 
                (1 - alpha) * frame[y:y+h, x:x+w, c]
            )
    else:
        frame[y:y+h, x:x+w] = symbol_np[:, :, :3]
    
    return frame

def generate_video(output_path='output_video.mp4', symbols_folder='symbols'):
    """Generate the video with specified parameters."""
    
    # Load symbols
    print("Loading symbols...")
    symbols = load_symbols(symbols_folder)
    if not symbols:
        raise ValueError(f"No symbols found in folder: {symbols_folder}")
    print(f"Loaded {len(symbols)} symbols")
    
    # Calculate total frames
    total_frames = VIDEO_DURATION * FPS
    center = (WIDTH // 2, HEIGHT // 2)
    
    # Generate random symbol appearances
    # Ensure no overlap by sorting by start time
    symbol_events = []
    symbol_duration_frames = int(SYMBOL_DURATION * FPS)
    
    # Generate random non-overlapping time points
    for _ in range(NUM_SYMBOLS):
        while True:
            start_frame = random.randint(0, total_frames - symbol_duration_frames)
            end_frame = start_frame + symbol_duration_frames
            
            # Check for overlap
            overlap = False
            for existing in symbol_events:
                if not (end_frame <= existing['start'] or start_frame >= existing['end']):
                    overlap = True
                    break
            
            if not overlap:
                symbol_events.append({
                    'start': start_frame,
                    'end': end_frame,
                    'symbol': random.choice(symbols).copy(),
                    'position': calculate_position(
                        random.choice(ANGLES),
                        random.choice(DISTANCES),
                        center
                    ),
                    'rotation': random.randint(0, 360),
                    'size': random.randint(80, 120)
                })
                break
    
    # Sort events by start time
    symbol_events.sort(key=lambda x: x['start'])
    
    print(f"Generated {len(symbol_events)} symbol events")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    print("Generating video frames...")
    current_event_idx = 0
    
    for frame_num in range(total_frames):
        # Create blank white frame
        frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
    
        # Draw cross at center
        cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (0, 0, 0), 2)
        cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (0, 0, 0), 2)
    
        
        # Check if any symbol should be displayed
        if current_event_idx < len(symbol_events):
            event = symbol_events[current_event_idx]
            
            if frame_num >= event['start'] and frame_num < event['end']:
                # Prepare symbol
                symbol = resize_and_rotate_symbol(
                    event['symbol'].copy(),
                    event['size'],
                    event['rotation']
                )
                
                # Overlay symbol
                frame = overlay_symbol(frame, symbol, event['position'])
            
            # Move to next event if current one is finished
            if frame_num >= event['end']:
                current_event_idx += 1
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_num % (FPS * 5) == 0:
            print(f"Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    generate_video(
        output_path='symbol_dataset.mp4',
        symbols_folder='symbols'
    )