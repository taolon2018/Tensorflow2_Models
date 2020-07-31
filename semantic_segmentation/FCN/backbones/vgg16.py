import tensorflow as tf


# structure reference: https://neurohive.io/en/popular-networks/vgg16/
class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv1_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="valid"
        )

        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="valid"
        )

        self.conv3_1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_2 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_3 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool3 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="valid"
        )

        self.conv4_1 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_2 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_3 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool4 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="valid"
        )

        self.conv5_1 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv5_2 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv5_3 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool5 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="valid"
        )

    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        pool3 = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        pool4 = x

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        pool5 = x

        return pool3, pool4, pool5
