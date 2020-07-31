import os
import random
import numpy as np
import cv2

voc_colormap = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def DataGenerator(voc_path, batch_size, split):
    """
    generate voc image and mask batch
    """
    assert split in ["test", "train", "val"]
    label_path = os.path.join(voc_path, "ImageSets", "Segmentation", split + ".txt")
    with open(label_path) as f:
        lines = f.readlines()
    random.shuffle(lines)

    def extract_raw_image(image_name):
        image_path = os.path.join(voc_path, "JPEGImages", image_name + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def extract_mask(image_name):
        mask_path = os.path.join(voc_path, "SegmentationClass", image_name + ".png")
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(mask_image.shape[:-1], dtype=np.int32)
        for x, y in np.ndindex(mask_image.shape[:-1]):
            point = mask_image[x, y].tolist()
            if point not in voc_colormap:
                mask[x, y] = 0
            else:
                mask[x, y] = voc_colormap.index(point)
        return mask

    annotation_len = len(lines)
    for i in range(0, annotation_len, batch_size):
        batch_lines = lines[i : min(annotation_len, i + batch_size)]
        image_names = [line.strip() for line in batch_lines]
        x = [extract_raw_image(image_name) for image_name in image_names]
        y = [extract_mask(image_name) for image_name in image_names]
        x, y = (
            np.stack(x, axis=0).astype(dtype=np.float32),
            np.stack(y, axis=0).astype(dtype=np.float32),
        )
        yield x, y
