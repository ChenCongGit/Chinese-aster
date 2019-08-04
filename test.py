import tensorflow as tf
import numpy as np
from scipy.misc import imrotate
import os
from PIL import Image


# def random_rotate(image, max_angle=5):
#   """
#   随机旋转一定的角度
#   Args:
#     image: Rank 3 float32 tensor containing 1 image -> [height, width, 3]
#            with pixel values varying between [0, 1].
#     max_angle: 最大旋转角度

#   Returns:
#     image: A single channel grayscale image -> [image, height, 1].
#   """
#   with tf.name_scope('RandomRotate', values=[image]):
#     def random_rotate_func(image):
#       #旋转角度范围
#       angle = np.random.uniform(low=-max_angle, high=max_angle)
#       return misc.imrotate(image, angle, 'bilinear')

#     rotated_image = tf.py_func(random_rotate_func, [image], tf.float32)
#     return rotated_image


# def main():
#     image_dir = ''
#     for dir in os.listdir(image_dir):
#         Image.open(os.path.join(image_dir, dir))


def random_rotate_image(image_file, num):
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=3)
        image_rotate_en_list = []
        def random_rotate_image_func(image):
            #旋转角度范围
            angle = np.random.uniform(low=-5.0, high=5.0)
            return imrotate(image, angle, 'bicubic')

        for i in range(num):
            image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8)
            image = tf.to_float(image_rotate)
            image_rotate_en_list.append(tf.image.encode_jpeg(image_rotate))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            results = sess.run(image_rotate_en_list)
            for idx,re in enumerate(results):
                with open('data/'+str(idx)+'.jpg','wb') as f:
                    f.write(re)


if __name__ == '__main__':
    #处理图片，进行20次随机处理，并将处理后的图片保存到输入图片相同的路径下
    random_rotate_image('data/test.jpg', 20)