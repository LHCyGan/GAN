#-*-coding:utf-8-*-
import tensorflow as tf
import glob
import numpy as np
import os
from scipy.misc import toimage
from dcgan.gan import Generator, Discriminator

from wgan.dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):

        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def celoss_ones(logits):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discrimator, batch_x, batch_z, training):
    fake_image = generator(batch_z, training)

    d_fake_logits = discrimator(fake_image, training)
    d_real_logits = discrimator(batch_x, training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_fake + d_loss_real
    return loss

def g_loss_fn(generator, discrimator, batch_z, training):

    fake_image = generator(batch_z)
    d_fake_logits = discrimator(fake_image, training)
    d_loss_fake = celoss_ones(d_fake_logits)

    return d_loss_fake




def main():
    tf.random.set_seed(998)
    np.random.seed(20)

    z_dim = 100
    epochs = 3000000
    batch_size = 128
    learning_rate = 0.002
    training = True

    img_path = glob.glob("./faces/*.jpg")

    dataset, image_shape, _ = make_anime_dataset(img_path, batch_size)
    sample = next(iter(dataset))
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discrimator = Discriminator()
    discrimator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    d_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(iter(db_iter))

        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discrimator, batch_x, batch_z, training)
        grads = tape.gradient(d_loss, discrimator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discrimator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discrimator, batch_z, training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epochs, " epoch d_loss:", float(d_loss), 'g_loss:', float(g_loss))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            if not os.path.exists("./images"):
                os.makedirs("./images")
            img_path = os.path.join('images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='p')


if __name__ == '__main__':
    main()