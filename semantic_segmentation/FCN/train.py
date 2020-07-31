import os

import tensorflow as tf
import argparse
from semantic_segmentation.FCN.model import FCN8

from semantic_segmentation.FCN.utils import DataGenerator

print(tf.__version__)


tensor_board_log_dir = "./log"


def train(args):
    if not os.path.exists(tensor_board_log_dir):
        os.mkdir(tensor_board_log_dir)
    writer = tf.summary.create_file_writer(tensor_board_log_dir)

    model = FCN8()

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    val_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for i in range(args.epoch):
        for it, (x, y) in enumerate(
            DataGenerator(
                voc_path=args.voc_path, batch_size=args.batch_size, split="train"
            )
        ):
            with tf.GradientTape() as tape:
                pred_y = model(x)
                loss = loss_func(y_true=y, y_pred=pred_y)
            train_loss_metrics(y_true=y, y_pred=pred_y)
            train_accuracy_metrics(y_true=y, y_pred=pred_y)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(
                f"==> {i + 1}th epoch, {it + 1}th iteration, {train_loss_metrics.result():.4f} train_loss, {train_accuracy_metrics.result() * 100:.4f}% train_accuracy"
            )

        for it, (x, y) in enumerate(
            DataGenerator(
                voc_path=args.voc_path, batch_size=args.batch_size, split="val"
            )
        ):
            pred_y = model(x)
            val_loss_metrics(y_true=y, y_pred=pred_y)
            val_accuracy_metrics(y_true=y, y_pred=pred_y)

        print(
            f"==> {i + 1}th epoch, {val_loss_metrics.result():.4f} val_loss, {val_accuracy_metrics.result() * 100 :.4f}% val_accuracy"
        )

        with writer.as_default():
            tf.summary.scalar("train_loss", train_loss_metrics.result(), step=i)
            tf.summary.scalar("train_accuracy", train_accuracy_metrics.result(), step=i)
            tf.summary.scalar("val_loss", val_loss_metrics.result(), step=i)
            tf.summary.scalar("val_accuracy", val_accuracy_metrics.result(), step=i)
        writer.flush()

        train_loss_metrics.reset_states()
        train_accuracy_metrics.reset_states()
        val_loss_metrics.reset_states()
        val_accuracy_metrics.reset_states()

        if not os.path.exists("./model_weight"):
            os.mkdir("./model_weight")
        model.save_weights("./model_weight/model.h5")


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
