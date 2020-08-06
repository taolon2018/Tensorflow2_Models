import argparse

from object_detection.YOLO_v3.model.yolov3 import Yolov3
import numpy as np


def train(args):
    model = Yolov3()
    i1, i2, i3 = model(np.zeros((8, 224, 224, 3), dtype=np.float))
    print(i1.shape, i2.shape, i3.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--voc_path",
        type=str,
        default="/home/taolong/dataset/VOC/train/VOCdevkit/VOC2007",
    )
    args = parser.parse_args()
    train(args)
