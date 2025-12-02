import cv2
import numpy as np

ANGLES = [45, 135, 225, 315]
DISTANCES = [200, 400, 600]


def create_2d_gaussian_mask(image_shape, center, sigma_x, sigma_y, angle=0):
    """
    Create a 2D Gaussian mask with elliptical shape.

    Parameters:
    - image_shape: tuple (height, width) of the image
    - center: tuple (y, x) center position of the gaussian
    - sigma_x: standard deviation along x-axis (controls width)
    - sigma_y: standard deviation along y-axis (controls height)
    - angle: rotation angle in degrees (optional)

    Returns:
    - mask: 2D array with values from 0 to 1
    """
    h, w = image_shape
    y, x = np.ogrid[0:h, 0:w]

    # Center coordinates
    y0, x0 = center

    # Convert angle to radians
    theta = np.radians(angle)

    # Rotation matrix components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotate coordinates
    x_rot = (x - x0) * cos_theta + (y - y0) * sin_theta
    y_rot = -(x - x0) * sin_theta + (y - y0) * cos_theta

    # 2D Gaussian formula
    mask = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

    return mask


def mask_frame(frame):
    """
    Apply masking to a single frame.
    Replace this with your actual masking logic.
    """
    center = (frame.shape[0] // 2, frame.shape[1] // 2)
    gauss_center = center
    gauss_angle = 0
    gauss_dist_x = 50
    gauss_dist_y = 50
    for a in ANGLES:
        got_gauss = False
        for d in DISTANCES:
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

    mask = create_2d_gaussian_mask(
        image_shape=frame.shape[:2],
        center=gauss_center,
        sigma_x=gauss_dist_x,
        sigma_y=gauss_dist_y,
        angle=gauss_angle,  # optional rotation
    )

    # Apply mask to image
    masked_img = frame.copy()
    for i in range(3):  # Apply to each RGB channel
        masked_img[:, :, i] = frame[:, :, i] * mask

    return masked_img


def main(input_video_path, output_video_path):
    """
    Read video frame by frame, apply masking, and save as new video.
    """
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply masking to frame
        masked_frame = mask_frame(frame)

        # Write masked frame to output video
        out.write(masked_frame)

        i += 1
        if i % 300 == 0:
            print("Progress:", i)

    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    input_video = "symbol_dataset.mp4"
    output_video = "output.mp4"
    main(input_video, output_video)
