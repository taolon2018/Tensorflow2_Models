import tensorflow as tf


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_size, padding="same", stride=1):
        super(ConvLayer, self).__init__()
        self.zero_padding = None
        if padding != "same":
            self.zero_padding = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.conv = tf.keras.layers.Conv2D(
            output_dim, kernel_size, padding=padding, strides=stride
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def __call__(self, x):
        if self.zero_padding:
            x = self.zero_padding(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ResBlock, self).__init__()
        self.layer_1 = ConvLayer(output_dim / 2, 1)
        self.layer_2 = ConvLayer(output_dim, 3)

    def __call__(self, x):
        input = x
        x = self.layer_1(x)
        x = self.layer_2(x)
        return input + x


class Darknet53(tf.keras.Model):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv_1 = ConvLayer(32, 3)
        self.conv_2 = ConvLayer(64, 3, padding="valid", stride=2)

        self.resblocks_1 = [ResBlock(64) for i in range(1)]
        self.conv_3 = ConvLayer(128, 3, padding="valid", stride=2)

        self.resblocks_2 = [ResBlock(128) for i in range(2)]
        self.conv_4 = ConvLayer(256, 3, padding="valid", stride=2)

        self.resblocks_3 = [ResBlock(256) for i in range(8)]
        self.conv_5 = ConvLayer(512, 3, padding="valid", stride=2)

        self.resblocks_4 = [ResBlock(512) for i in range(8)]
        self.conv_6 = ConvLayer(1024, 3, padding="valid", stride=2)

        self.resblocks_5 = [ResBlock(1024) for i in range(4)]

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        for resblock in self.resblocks_1:
            x = resblock(x)
        x = self.conv_3(x)

        for resblock in self.resblocks_2:
            x = resblock(x)
        x = self.conv_4(x)

        for resblock in self.resblocks_3:
            x = resblock(x)
        output1 = x
        x = self.conv_5(x)

        for resblock in self.resblocks_4:
            x = resblock(x)
        output2 = x
        x = self.conv_6(x)

        for resblock in self.resblocks_5:
            x = resblock(x)
        return output1, output2, x
