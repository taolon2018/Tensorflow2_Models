import tensorflow as tf

from semantic_segmentation.FCN.backbones.vgg16 import VGG16


class FCN8(tf.keras.Model):
    def __init__(self, output_dim=21):
        super(FCN8, self).__init__()
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(100, 100))
        self.vgg16 = VGG16()

        self.fc6 = tf.keras.layers.Conv2D(
            filters=4096, kernel_size=(7, 7), padding="valid", activation="relu"
        )
        self.dropout6 = tf.keras.layers.Dropout(rate=0.5)

        self.fc7 = tf.keras.layers.Conv2D(
            filters=4096, kernel_size=(1, 1), padding="valid", activation="relu"
        )
        self.dropout7 = tf.keras.layers.Dropout(rate=0.5)

        self.score_fc7 = tf.keras.layers.Conv2D(
            filters=output_dim, kernel_size=(1, 1), padding="valid"
        )
        self.score_pool4 = tf.keras.layers.Conv2D(
            filters=output_dim, kernel_size=(1, 1), padding="valid"
        )
        self.score_pool3 = tf.keras.layers.Conv2D(
            filters=output_dim, kernel_size=(1, 1), padding="valid"
        )

        self.up_sampling_1 = tf.keras.layers.Conv2DTranspose(
            filters=output_dim,
            kernel_size=(4, 4),
            strides=2,
            use_bias=False,
            padding="valid",
        )
        self.up_sampling_2 = tf.keras.layers.Conv2DTranspose(
            filters=output_dim,
            kernel_size=(4, 4),
            strides=2,
            use_bias=False,
            padding="valid",
        )
        self.up_sampling_3 = tf.keras.layers.Conv2DTranspose(
            filters=output_dim,
            kernel_size=(16, 16),
            strides=8,
            use_bias=False,
            padding="valid",
        )

    def call(self, x):
        input = x
        x = self.zero_padding(x)
        pool3, pool4, pool5 = self.vgg16(x)
        x = self.fc6(pool5)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.dropout7(x)
        x = self.score_fc7(x)

        x = self.up_sampling_1(x)
        x = x + self.score_pool4(pool4)[:, 5 : 5 + x.shape[1], 5 : 5 + x.shape[2], :]

        x = self.up_sampling_2(x)
        x = x + self.score_pool3(pool3)[:, 9 : 9 + x.shape[1], 9 : 9 + x.shape[2], :]

        x = self.up_sampling_3(x)[
            :, 31 : 31 + input.shape[1], 31 : 31 + input.shape[2], :
        ]

        return x
