import argparse
import os

from semantic_segmentation.FCN.model import FCN8
from semantic_segmentation.FCN.utils import DataGenerator, voc_colormap
import tensorflow as tf
import numpy as np
from PIL import Image


def mask_reconstruct_concatenate_save(pred_y, y, it):
    concatenated_tensor = tf.cast(tf.concat([pred_y, y], axis=1), dtype=tf.int32)
    img = np.zeros(shape=[224, 448, 3], dtype=np.int8)
    for a1 in range(concatenated_tensor.shape[0]):
        for a2 in range(concatenated_tensor.shape[1]):
            color_map_idx = concatenated_tensor[a1, a2]
            img[a1, a2] = np.array(voc_colormap[color_map_idx])
    img = Image.fromarray(img, "RGB")
    if not os.path.exists("./prediction_result"):
        os.mkdir("./prediction_result")
    img.save(f"./prediction_result/{it}.jpg")
    print(f"wrote ./prediction_result/{it}.jpg")


def test(args):
    model = FCN8()
    model.build((1, 255, 255, 3))
    model.load_weights("./model_weight/model.h5")
    test_loss_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    for it, (x, y) in enumerate(
        DataGenerator(voc_path=args.voc_path, batch_size=args.batch_size, split="test")
    ):
        pred_y = model(x)
        test_loss_metrics(y_true=y, y_pred=pred_y)
        test_accuracy_metrics(y_true=y, y_pred=pred_y)
        pred_y = tf.argmax(pred_y, axis=-1)
        mask_reconstruct_concatenate_save(pred_y[0], y[0], it)
    print(
        f"{test_loss_metrics.result():.4f} train_loss, {test_accuracy_metrics.result() * 100:.4f}% train_accuracy"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--voc_path",
        type=str,
        default="/home/taolong/dataset/VOC/test/VOCdevkit/VOC2007",
    )
    args = parser.parse_args()
    test(args)
