# -*- coding: utf-8 -*-

import os
import io
import random
import re
import glob
import shutil
import numpy as np

from PIL import Image
import tensorflow as tf

from Chinese_aster.utils import dataset_util
from Chinese_aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/AI/chencong/Chinese_aster/ocr_dataset/new_ocr_train/', 'Root directory to raw SynthText dataset.')
flags.DEFINE_bool('exclude_difficult', False, 'Excluding non-alphanumeric examples.')
flags.DEFINE_string('output_path', '/home/AI/chencong/Chinese_aster/ocr_dataset/', 'Output tfrecord path.')
flags.DEFINE_bool('padding', True, 'padding image to same aspect ratio')
flags.DEFINE_integer('pad_threshold', 8, 'if aspect ratio is more than it, remain original image, else padding')
FLAGS = flags.FLAGS


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def char_check(word):
  if not word.isalnum():
    return False
  else:
    for char in word:
      if char < ' ' or char > '~':
        return False
  return True


def padding_image(img):
  width = img.size[0]
  height = img.size[1]
  if width/height >= FLAGS.pad_threshold:
    return img
  else:
    image_array = np.asarray(img)
    pixel = np.mean(image_array)
    max_pixel = np.max(image_array)
    min_pixel = np.min(image_array)
    pad_pixel = min(int(pixel + (max_pixel - min_pixel)/6), 255)

    width_left_pad = int((height * FLAGS.pad_threshold - width)/2)
    width_right_pad = int(height * FLAGS.pad_threshold - width - width_left_pad)
    image_array = np.pad(image_array, ((0,0),(width_left_pad, width_right_pad)), 'constant', constant_values=(pad_pixel,pad_pixel))
    img = Image.fromarray(image_array)
    return img


def create_ic15(output_path):

  groundtruth_file_path = os.path.join(FLAGS.data_dir, 'test_groundtruth_all.txt')
  
  with open(groundtruth_file_path, 'r') as f:
    lines = f.readlines()
    img_gts = [line.strip() for line in lines]
    a_test = 0
    
    writer = tf.python_io.TFRecordWriter(output_path + 'ocr_train.tfrecord')

    for img_gt in img_gts:
      create_tfrecord(img_gt, writer)
      a_test += 1
    writer.close()
    print('a_test: ', a_test)


def create_tfrecord(img_gt, writer):
  
  # print(img_gt)
  img_rel_path, gt = img_gt.split(', ',1)

  new_gt = ""
  for one_char in gt:
    new_char = one_char + " "
    new_gt += new_char
  new_gt = new_gt[:-1]

  img_path = os.path.join(FLAGS.data_dir, img_rel_path)
  print(img_path, gt)
  img = Image.open(img_path).convert('L')
  if FLAGS.padding:
    img = padding_image(img)
    img.save('./Chinese_aster/tools/tmp_image/{}'.format(img_rel_path))

  img_buff = io.BytesIO()
  img.save(img_buff, format='jpeg')
  word_crop_jpeg = img_buff.getvalue()
  crop_name = os.path.basename(img_path)

  example = tf.train.Example(features=tf.train.Features(feature={
    fields.TfExampleFields.image_encoded: \
      dataset_util.bytes_feature(word_crop_jpeg),
    fields.TfExampleFields.image_format: \
      dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    fields.TfExampleFields.filename: \
      dataset_util.bytes_feature(crop_name.encode('utf-8')),
    fields.TfExampleFields.channels: \
      dataset_util.int64_feature(3),
    fields.TfExampleFields.colorspace: \
      dataset_util.bytes_feature('rgb'.encode('utf-8')),
    fields.TfExampleFields.transcript_gt_with_blank: \
      dataset_util.bytes_feature(new_gt.encode('utf-8')),
    fields.TfExampleFields.transcript_gt: \
      dataset_util.bytes_feature(gt.encode('utf-8')),
  }))
  writer.write(example.SerializeToString())


if __name__ == '__main__':
  create_ic15(FLAGS.output_path)
