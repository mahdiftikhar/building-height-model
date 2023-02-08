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
