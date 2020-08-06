import tensorflow as tf

from object_detection.YOLO_v3.backbone.darknet53 import Darknet53, ConvLayer


class ConvSet(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ConvSet, self).__init__()
        self.conv_1 = ConvLayer(output_dim, 1)
        self.conv_2 = ConvLayer(output_dim * 2, 3)
        self.conv_3 = ConvLayer(output_dim, 1)
        self.conv_4 = ConvLayer(output_dim * 2, 3)
        self.conv_5 = ConvLayer(output_dim, 1)

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x


class Yolov3(tf.keras.Model):
    def __init__(self, predict_class_number=21):
        super(Yolov3, self).__init__()
        self.darknet53 = Darknet53()
        self.convset_1 = ConvSet(512)
        self.small_branch_conv_1 = ConvLayer(1024, 1)
        self.small_branch_conv_2 = tf.keras.layers.Conv2D(
            3 * (predict_class_number + 5), 1, activation=None
        )
        self.conv_1 = ConvLayer(256, 1)
        self.convset_2 = ConvSet(256)
        self.medium_branch_conv_1 = ConvLayer(512, 1)
        self.medium_branch_conv_2 = tf.keras.layers.Conv2D(
            3 * (predict_class_number + 5), 1, activation=None
        )
        self.conv_2 = ConvLayer(512, 1)
        self.convset_3 = ConvSet(128)
        self.large_branch_conv_1 = ConvLayer(256, 1)
        self.large_branch_conv_2 = tf.keras.layers.Conv2D(
            3 * (predict_class_number + 5), 1, activation=None
        )
        self.conv_3 = ConvLayer(1024, 1)

    def __call__(self, x):
        input_1, input_2, input_3 = self.darknet53(x)
        x = input_3

        x = self.convset_1(x)

        output_1 = self.small_branch_conv_1(x)
        output_1 = self.small_branch_conv_2(output_1)

        x = self.conv_1(x)
        x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method="nearest")
        x = tf.concat([x, input_2], axis=-1)

        x = self.convset_2(x)

        output_2 = self.medium_branch_conv_1(x)
        output_2 = self.medium_branch_conv_2(output_2)

        x = self.conv_2(x)
        x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method="nearest")
        x = tf.concat([x, input_1], axis=-1)

        x = self.convset_3(x)

        output_3 = self.large_branch_conv_1(x)
        output_3 = self.large_branch_conv_2(output_3)

        return output_1, output_2, output_3
