from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils.external.common import Reader

from . import _abstract

__all__ = ['parse_instances_annotations']


def parse_instances_annotations(path: str) -> Tuple[Dict[int, Dict],
                                                    Dict[int, List[Dict[str, Any]]]]:

    origin = Reader.json_to_dict(path)

    categories = {}
    annotations = {}

    for ann in tqdm(origin['annotations']):

        image_id = ann.pop('image_id')

        if image_id not in annotations:

            annotations[image_id] = []

        annotations[image_id].append(ann)

    for ann in tqdm(origin['images']):

        image_id = ann.pop('id')

        if image_id in annotations:

            annotations[image_id].append(ann)

    for cat in origin['categories']:

        category_id = cat.pop('id')

        categories[category_id] = cat

    return categories, annotations


class Annotation(_abstract.Meta):

    def __init__(self, path: str, image_size: Optional[Tuple[int, int]] = None):

        self.path = path

        self.categories, self.annotations = parse_instances_annotations(self.path)

        self.__image_size__ = image_size

    def num_objects_by_id(self, image_id):

        return len(self.annotations[image_id][:-1])

    def image_size_by_id(self, image_id):

        meta = self.annotations[image_id][-1]

        image_size = (meta['height'], meta['width'])

        return image_size

    def categories_by_id(self, image_id):

        num_objects = self.num_objects_by_id(image_id)

        categories = np.zeros((num_objects, ))

        for i, ann in enumerate(self.annotations[image_id][:-1]):

            categories[i] = ann['category_id']

        return categories

    def boxes_by_id(self, image_id):

        num_objects = self.num_objects_by_id(image_id)

        image_size = self.image_size_by_id(image_id)

        boxes = np.zeros((num_objects, 4))

        for i, ann in enumerate(self.annotations[image_id][:-1]):

            boxes[i] = ann['bbox']

        if self.image_size is not None:

            boxes = _abstract.points_resized(boxes, image_size, self.image_size)

        return boxes

    def polygons_by_id(self, image_id):

        image_size = self.image_size_by_id(image_id)

        polygons = []

        for ann in self.annotations[image_id][:-1]:

            polygon = np.asarray(ann['segmentation'], dtype=np.float32)

            if self.image_size is not None:

                polygon = _abstract.points_resized(polygon, image_size, self.image_size)

            polygons.append(polygon)

        return polygons

    def masks_by_id(self, image_id, mask_size: Tuple = (20, 20),
                    interpolation=cv2.INTER_AREA, padding: int = 4):

        num_objects = self.num_objects_by_id(image_id)

        image_size = self.image_size_by_id(image_id)

        polygons = self.polygons_by_id(image_id)
        boxes = self.boxes_by_id(image_id)

        h, w = mask_size

        masks = np.zeros((h + 2 * padding, w + 2 * padding, num_objects))

        for i in range(num_objects):

            mask = _abstract.polygon_to_mask(polygons[i], image_size, image_size, color=1)

            masks[..., i] = \
                _abstract.mask_crop_and_resize(mask, mask_size, boxes[i], interpolation=interpolation,
                                               padding=padding)

        return masks.round().astype('uint8')

    def is_crowd_by_id(self, image_id):

        num_objects = self.num_objects_by_id(image_id)

        is_crowd = np.zeros((num_objects, ), dtype=np.bool)

        for i, ann in enumerate(self.annotations[image_id][:-1]):

            is_crowd[i] = ann['iscrowd']

        return is_crowd

    def overlap_counts_by_id(self, image_id, grid_size):

        num_objects = self.num_objects_by_id(image_id)

        if self.image_size is None:

            image_size = self.image_size_by_id(image_id)

        else:

            image_size = self.image_size

        overlaps = np.zeros(grid_size, dtype=np.int32)
        boxes = self.boxes_by_id(image_id)

        for k in range(num_objects):

            i, j = _abstract.bbox_to_loc(boxes[k], image_size, grid_size)

            overlaps[i, j] += 1

        return overlaps - 1

    def max_num_objects(self):

        counts = 0

        for image_id in tqdm(self.annotations):

            counts = max(counts, self.num_objects_by_id(image_id))

        return counts

    def max_overlaps(self, grid_size):

        counts = 0

        for image_id in tqdm(self.annotations):

            counts = max(counts, self.overlap_counts_by_id(image_id, grid_size).max())

        return counts

    @property
    def image_size(self):

        return self.__image_size__

    def reinterpret_size(self, image_size):

        self.__image_size__ = image_size
