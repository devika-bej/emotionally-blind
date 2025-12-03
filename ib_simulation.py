import cv2
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms

from Hebbian_model import ImprovedHebbianFear

pos_angles = [45, 135, 225, 315]
pos_dists = [200, 400, 600]

current_verdict = "NEUTRAL"
final_verdict = ""
neutral_detected = False
detected_frame_countdown = -1
identified_frame_countdown = -1


def get_roi_patch(frame, center, size=128):
    x, y = int(center[0]), int(center[1])
    r = size // 2
    h, w, _ = frame.shape
    padded = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    x_pad, y_pad = x + r, y + r
    patch = padded[x_pad - r : x_pad + r, y_pad - r : y_pad + r]
    return patch


def update_verdict(masked_frame, center):
    global current_verdict, final_verdict
    model.eval()
    torch.no_grad()
    patch = get_roi_patch(masked_frame, center)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = Image.fromarray(patch)
    fc_inp = preprocess(patch)
    fear_score = model(fc_inp)
    print(fear_score)
    if fear_score < 0.3:
        current_verdict = "NEUTRAL"
    elif fear_score < 0.8:
        current_verdict = "DETECTED"
    else:
        current_verdict = "IDENTIFIED"
    final_verdict = current_verdict
    print(f"Stimulus was of category: {current_verdict}")


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


def mask_frame(frame):
    center = (frame.shape[0] // 2, frame.shape[1] // 2)
    gauss_center = center
    gauss_angle = 0
    gauss_dist_x = 50
    gauss_dist_y = 50
    for a in pos_angles:
        got_gauss = False
        for d in pos_dists:
            a_rad = np.radians(a)
            x = center[0] + int(d * np.cos(a_rad))
            y = center[1] + int(d * np.sin(a_rad))
            patch = frame[x - 10 : x + 10, y - 10 : y + 10]
            for i in patch:
                for j in i:
                    if j[0] == 0:
                        got_gauss = True
                        break
                if got_gauss:
                    break
            if got_gauss:
                gauss_center = ((center[0] + x) / 2, (center[1] + y) / 2)
                gauss_angle = a
                gauss_dist_x = np.abs(x - gauss_center[0])
                gauss_dist_y = np.abs(y - gauss_center[1])
                break
        if got_gauss:
            break

    global current_verdict, neutral_detected, detected_frame_countdown, identified_frame_countdown

    # no unexpected stimulus so far
    if current_verdict == "NEUTRAL":
        # unexpected stimulus caught
        if got_gauss and not neutral_detected:
            mask = create_2d_gaussian_mask(
                image_shape=frame.shape[:2],
                center=gauss_center,
                sigma_x=gauss_dist_x,
                sigma_y=gauss_dist_y,
                angle=gauss_angle,
            )
            masked_img = frame.copy()
            for i in range(3):
                masked_img[:, :, i] = frame[:, :, i] * mask

            # check if fearful
            update_verdict(masked_img, gauss_center)
            # not fearful enough
            if current_verdict == "DETECTED":
                detected_frame_countdown = 5
            if current_verdict == "NEUTRAL":
                neutral_detected = True
        # no unexpected stimulus caught or a neutral stimulus is present
        else:
            mask = create_2d_gaussian_mask(
                image_shape=frame.shape[:2],
                center=center,
                sigma_x=50,
                sigma_y=50,
                angle=0,
            )
            masked_img = frame.copy()
            for i in range(3):
                masked_img[:, :, i] = frame[:, :, i] * mask
    # had found a slightly fearful stimulus
    elif current_verdict == "DETECTED":
        # still in thought
        if detected_frame_countdown < 0:
            mask = create_2d_gaussian_mask(
                image_shape=frame.shape[:2],
                center=center,
                sigma_x=50,
                sigma_y=50,
                angle=0,
            )
            masked_img = frame.copy()
            for i in range(3):
                masked_img[:, :, i] = frame[:, :, i] * mask
            if not got_gauss:
                current_verdict = "NEUTRAL"
        # no more in thought
        else:
            mask = create_2d_gaussian_mask(
                image_shape=frame.shape[:2],
                center=gauss_center,
                sigma_x=gauss_dist_x,
                sigma_y=gauss_dist_y,
                angle=gauss_angle,
            )
            masked_img = frame.copy()
            for i in range(3):
                masked_img[:, :, i] = frame[:, :, i] * mask
            detected_frame_countdown -= 1
    # highly fearful stimulus was found
    elif current_verdict == "IDENTIFIED":
        # still there on screen so alert
        if got_gauss:
            identified_frame_countdown = 10
            mask = create_2d_gaussian_mask(
                image_shape=frame.shape[:2],
                center=gauss_center,
                sigma_x=gauss_dist_x,
                sigma_y=gauss_dist_y,
                angle=gauss_angle,
            )
            masked_img = frame.copy()
            for i in range(3):
                masked_img[:, :, i] = frame[:, :, i] * mask
        # no more on screen
        else:
            # no more alert about stimulus
            if identified_frame_countdown < 0:
                mask = create_2d_gaussian_mask(
                    image_shape=frame.shape[:2],
                    center=center,
                    sigma_x=50,
                    sigma_y=50,
                    angle=0,
                )
                masked_img = frame.copy()
                for i in range(3):
                    masked_img[:, :, i] = frame[:, :, i] * mask
                current_verdict = "NEUTRAL"
            # still alert because highly fearful
            else:
                mask = create_2d_gaussian_mask(
                    image_shape=frame.shape[:2],
                    center=gauss_center,
                    sigma_x=gauss_dist_x,
                    sigma_y=gauss_dist_y,
                    angle=gauss_angle,
                )
                masked_img = frame.copy()
                for i in range(3):
                    masked_img[:, :, i] = frame[:, :, i] * mask
                identified_frame_countdown -= 1

    return masked_img


def main(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        masked_frame = mask_frame(frame)
        out.write(masked_frame)

        i += 1
        if i % 300 == 0:
            print("Progress:", i)

    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    global model, preprocess
    model = ImprovedHebbianFear(64 * 64)
    model.load_state_dict(torch.load("./fear_model_1080p(1).pth"))
    preprocess = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
        ]
    )

    input_dir = "ib_dataset"
    output_dir = "ib_dataset_output"
    results_file = "results.txt"
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}

    with open(results_file, "w") as f:
        for fname in sorted(os.listdir(input_dir)):
            if not any(fname.lower().endswith(ext) for ext in valid_exts):
                continue
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            print(f"Processing {in_path} -> {out_path}")
            try:
                main(in_path, out_path)
                f.write(f"{fname}: {final_verdict}\n")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
