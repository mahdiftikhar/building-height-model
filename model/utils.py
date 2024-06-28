import cv2
import numpy as np
from pvlib.solarposition import get_solarposition


def its_denormalize_time(bbox, image_shape=(640, 640)):
    if bbox.ndim == 1:
        denorm_bbox = bbox.copy().reshape(1, -1)
    else:
        denorm_bbox = bbox.copy()

    denorm_bbox[:, 0] *= image_shape[0]
    denorm_bbox[:, 2] *= image_shape[0]
    denorm_bbox[:, 1] *= image_shape[1]
    denorm_bbox[:, 3] *= image_shape[1]

    if bbox.ndim == 1:
        denorm_bbox = denorm_bbox[0]

    return denorm_bbox


def its_xyxy_time(bbox):
    num_dims = bbox.ndim

    if num_dims == 1:
        xyxy = bbox.copy().reshape(1, -1)
        bbox = bbox.copy().reshape(1, -1)
    else:
        xyxy = bbox.copy()
    cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    xyxy[:, 0] = cx - (w / 2)
    xyxy[:, 1] = cy - (h / 2)
    xyxy[:, 2] = cx + (w / 2)
    xyxy[:, 3] = cy + (h / 2)

    if num_dims == 1:
        xyxy = xyxy[0]

    return xyxy.astype(int)


def get_solar_elevation(time, lat, long):
    """
    Lat and long are in degrees
    Time is datetime object

    Returns elevation in degrees
    """
    solpos = get_solarposition(time, lat, long)
    return solpos.elevation.values[0]

def resize_with_padding(img, target_size=(200, 200), padding_color=(114, 114, 114)):
    """
    Resize an image while maintaining aspect ratio and add padding to fill the empty space.

    :param image: input image.
    :param target_size: Tuple (width, height) of the target size.
    :param padding_color: Tuple (B, G, R) color value for padding. Default is white (255, 255, 255).
    """
    # Read the image
    original_height, original_width = img.shape[:2]

    # Calculate the ratio to maintain aspect ratio
    img_ratio = original_width / original_height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image is wider than the target ratio, fit to width
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller than the target ratio, fit to height
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and padding color
    padded_img = np.full((target_size[1], target_size[0], 3), padding_color, dtype=np.uint8)

    # Calculate the padding offsets
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    # Insert the resized image into the padded image
    padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
    return padded_img