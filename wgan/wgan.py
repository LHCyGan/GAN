#-*-coding:utf-8-*-
import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc = tf.keras.layers.Dense(3*3*512)

        self.conv1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), 3, 'valid')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):

        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x)))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = tf.nn.tanh(x)

        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), 3, padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), 3, padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(256, (5, 5), 3, padding='valid')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(1)


    def call(self, inputs, training=None):

        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x)))

        x = self.flatten(x)

        logits = self.fc(x)
        return logits

def main():

    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    print(d(x))

    print(g(z).shape)


if __name__ == '__main__':
    main()