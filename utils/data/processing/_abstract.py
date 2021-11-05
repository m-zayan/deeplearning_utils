import numpy as np

import cv2

import tensorflow as tf

from utils.ops.metrics import iou

__all__ = ['Meta']


def polygon_to_mask(polygon, image_size, mask_size, color=1, **kwargs):

    points = np.asarray(polygon, dtype=np.float32)
    points = points.reshape(-1, 2)

    scale_x = mask_size[1] / image_size[1]
    scale_y = mask_size[0] / image_size[0]

    points[:, 0] *= scale_x
    points[:, 1] *= scale_y

    points = points.round().astype(np.int32)

    mask = np.zeros(mask_size, dtype=np.int32)
    mask = cv2.fillPoly(mask, [points], color, **kwargs)

    return mask


def mask_to_bbox(mask):

    i, j = np.where(mask)

    x_min = j.min()
    x_max = j.max()

    y_min = i.min()
    y_max = i.max()

    width = x_max - x_min
    height = y_max - y_min

    bbox = np.array([x_min, y_min, width, height], dtype=np.float32)

    return bbox


def crop_mask(mask, bbox=None):

    if bbox is None:

        bbox = mask_to_bbox(mask)

    bbox = np.asarray(bbox, dtype=np.float32).round().astype('int32')

    cropped = tf.image.crop_to_bounding_box(mask[None, ..., None], bbox[1], bbox[0], bbox[3], bbox[2])

    return cropped.numpy().squeeze()


def crop_mask_inverse(cropped_mask, size, bbox):

    h, w = cropped_mask.shape

    coords = bbox_to_coords(bbox, standard=True)

    x_min, y_min, x_max, y_max = coords.round().astype('int32')

    if (y_max - y_min) != h:

        raise ValueError('...')

    if (x_max - x_min) != w:

        raise ValueError('...')

    mask = np.zeros(size, dtype=cropped_mask.dtype)

    mask[y_min:y_max, x_min:x_max] = cropped_mask

    return mask


def mask_crop_and_resize(mask, size, bbox=None, interpolation=cv2.INTER_NEAREST, padding: int = 0):

    cropped = crop_mask(mask, bbox)
    cropped = cropped.astype('float32')

    resized = cv2.resize(cropped, size, interpolation=interpolation)

    return np.pad(resized, padding)


def mask_crop_and_resize_inverse(resized_mask, size, bbox, interpolation=cv2.INTER_NEAREST, padding: int = 0):

    coords = bbox_to_coords(bbox, standard=True)

    x_min, y_min, x_max, y_max = coords.round().astype('int32')

    w = x_max - x_min
    h = y_max - y_min

    cropped = mask_padding_inverse(resized_mask, padding)

    cropped = cv2.resize(cropped, (w, h), interpolation=interpolation)

    mask = crop_mask_inverse(cropped, size, bbox)

    return mask


def mask_padding_inverse(mask, padding):

    h, w = mask.shape

    return mask[padding:h-padding, padding:w-padding]


def bbox_scale_normalization(bbox, image_size, standard=False):

    """ if standard is True, then, bbox = [x1, y1, x2, y2] """

    if standard:

        image_size = image_size[::-1]

    scale = np.array([*image_size, *image_size]) - 1.0

    normalized = np.divide(bbox, scale)

    return normalized.astype('float32')


def bbox_scale_denormalization(normalized, image_size, standard=False):

    """ if standard is True, then, bbox = [x1, y1, x2, y2] """

    if standard:

        image_size = image_size[::-1]

    scale = np.array([*image_size, *image_size]) - 1.0

    bbox = np.multiply(normalized, scale)

    return bbox.astype('float32')


def bbox_to_coords(bbox, standard=False):

    x, y, width, height = bbox

    x_min = x
    y_min = y

    x_max = x + width
    y_max = y + height

    if standard:

        coords = np.array([x_min, y_min, x_max, y_max])

    else:

        coords = np.array([y_min, x_min, y_max, x_max])

    return coords


def coords_to_bbox(coords, standard=False):

    if standard:

        x_min, y_min, x_max, y_max = coords

    else:

        y_min, x_min, y_max, x_max = coords

    width = x_max - x_min
    height = y_max - y_min

    bbox = np.array([x_min, y_min, width, height])

    return bbox


def bbox_center_coords(bbox):

    x, y, width, height = bbox

    x_offset = width / 2
    y_offset = height / 2

    x_center = x + x_offset
    y_center = y + y_offset

    center = np.array([y_center, x_center]).round()

    return center.astype('int32')


def compute_strides(image_size, grid_size):

    if image_size[0] < grid_size[0]:

        raise ValueError('...')

    if image_size[1] < grid_size[1]:

        raise ValueError('...')

    sy = np.round(image_size[0] / grid_size[0])
    sx = np.round(image_size[1] / grid_size[1])

    strides = np.array([sy, sx])

    return strides.astype('int32')


def bbox_to_loc(bbox, image_size, grid_size):

    center = bbox_center_coords(bbox)
    strides = compute_strides(image_size, grid_size)

    i = np.round(center[0] / strides[0]) - 1
    i = np.clip(i, a_min=0, a_max=grid_size[0] - 1)

    j = np.round(center[1] / strides[1])
    j = np.clip(j, a_min=0, a_max=grid_size[1] - 1)

    ij = np.asarray([i, j], dtype=np.int32)

    return ij


def points_resized(data, image_size, new_size):

    shape = data.shape

    scale_y = new_size[0] / image_size[0]
    scale_x = new_size[1] / image_size[1]
    scale = np.array([scale_x, scale_y])

    points = data.reshape(-1, 2)
    points = points * scale[None, :]

    return points.reshape(shape)


def match_bbox(bbox1, bbox2):

    coords1 = bbox_to_coords(bbox1)
    coords2 = bbox_to_coords(bbox2)

    score = iou(coords1, coords2)

    return score


class Meta:
    pass
