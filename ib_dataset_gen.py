import cv2
import numpy as np
import os
import random
from PIL import Image

video_length = 4
fps = 30
frame_w, frame_h = 1920, 1080
bg_color = (255, 255, 255)
symbol_time = 1

pos_angles = [45, 135, 225, 315]
pos_dists = [200, 400, 600]


def load_symbols(folder_path):
    symbols = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGBA")
            symbols.append(img)
    return symbols


def calculate_position(angle, distance, center):
    angle_rad = np.radians(angle)
    x = center[0] + int(distance * np.cos(angle_rad))
    y = center[1] + int(distance * np.sin(angle_rad))
    return (x, y)


def resize_and_rotate_symbol(symbol, size, rotation=0):
    symbol.thumbnail((size, size), Image.Resampling.LANCZOS)
    if rotation != 0:
        symbol = symbol.rotate(rotation, expand=True, fillcolor=(255, 255, 255))
    return symbol


def overlay_symbol(frame, symbol, position):
    symbol_np = np.array(symbol)
    h, w = symbol_np.shape[:2]
    x = position[0] - w // 2
    y = position[1] - h // 2
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    frame[y : y + h, x : x + w] = symbol_np[:, :, :3]
    return frame


def generate_video(output_path, symbol_content):
    total_frames = video_length * fps
    center = (frame_w // 2, frame_h // 2)
    symbol_duration_frames = int(symbol_time * fps)

    start_frame = random.randint(0, total_frames - symbol_duration_frames)
    end_frame = start_frame + symbol_duration_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    symbol_size = random.randint(80, 120)
    symbol_rot = random.randint(0, 360)
    symbol_pos = calculate_position(
        random.choice(pos_angles), random.choice(pos_dists), center
    )

    for frame_num in range(total_frames):
        frame = np.full((frame_h, frame_w, 3), bg_color, dtype=np.uint8)
        cv2.line(
            frame,
            (center[0] - 20, center[1]),
            (center[0] + 20, center[1]),
            (0, 0, 0),
            2,
        )
        cv2.line(
            frame,
            (center[0], center[1] - 20),
            (center[0], center[1] + 20),
            (0, 0, 0),
            2,
        )

        if frame_num >= start_frame and frame_num <= end_frame:
            symbol = resize_and_rotate_symbol(symbol_content, symbol_size, symbol_rot)
            frame = overlay_symbol(frame, symbol, symbol_pos)

        out.write(frame)

    out.release()
    print(output_path)


if __name__ == "__main__":
    symbols = load_symbols("./symbols")
    for filename, symbol in zip(os.listdir("./symbols"), symbols):
        sym_name = filename[:-4]
        for i in range(1, 31):
            generate_video(f"./ib_dataset/{sym_name}_{i}.mp4", symbol)
