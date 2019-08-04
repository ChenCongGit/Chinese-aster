import os
import re
import io
import logging
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
import numpy as np

from Chinese_aster.core import prefetcher
from Chinese_aster.protos import pipeline_pb2
from Chinese_aster.builders import model_builder
from Chinese_aster.builders import input_reader_builder
from Chinese_aster.core import standard_fields as fields

# supress TF logging duplicates
logging.getLogger('tensorflow').propagate = False
tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('exp_dir', 'Chinese_aster/experiments/demo/',
                    'Directory containing config, training log and evaluations')
flags.DEFINE_string('input_image_dir', '/home/AI/chencong/Chinese_aster/ocr_dataset/new_ocr_test/', 'Demo image')
flags.DEFINE_integer('check_num', 19029, 'choose one model to test')
flags.DEFINE_bool('padding', True, 'padding image to same aspect ratio')
flags.DEFINE_integer('pad_threshold', 8, 'if aspect ratio is more than it, remain original image, else padding')
FLAGS = flags.FLAGS


def get_configs_from_exp_dir():
  pipeline_config_path = os.path.join(FLAGS.exp_dir, 'config/ocr_eval.prototxt')

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader

  return model_config, eval_config, input_config


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


def main(_):
  # eval_dir = os.path.join(FLAGS.exp_dir, 'log/eval')
  model_config, _, _ = get_configs_from_exp_dir()

  model = model_builder.build(model_config, is_training=False)

  input_image_str_tensor = tf.placeholder(
    dtype=tf.string,
    shape=[])
  input_image_tensor = tf.image.decode_jpeg(
    input_image_str_tensor,
    channels=3,
  )
  resized_image_tensor = tf.image.resize_images(
    tf.to_float(input_image_tensor),
    [32, 256])

  resized_image_tensor_with_batch = tf.expand_dims(resized_image_tensor, 0)
  predictions_dict = model.predict(resized_image_tensor_with_batch)
  recognitions_list, recognitions_score_list = model.postprocess(predictions_dict)
  # recognition_text = recognitions['text'][0]
  # control_points = predictions_dict['control_points'],
  # rectified_images = predictions_dict['rectified_images']

  saver = tf.train.Saver(tf.global_variables())
  check_num = FLAGS.check_num
  checkpoint = os.path.join(FLAGS.exp_dir, 'log/model.ckpt-{}'.format(check_num))
  # checkpoint = os.path.join(FLAGS.exp_dir, 'log/model.ckpt')

  fetches = {
    'original_image': input_image_tensor,
    'recognitions': recognitions_list,
    'recognitions_score': recognitions_score_list
    # 'recognition_text': recognition_text,
    # 'control_points': predictions_dict['control_points'],
    # 'rectified_images': predictions_dict['rectified_images'],
  }

  f = open(FLAGS.input_image_dir+'test_groundtruth_all.txt')
  img_label_list = [line.split(',', 1) for line in f.readlines()]# if not re.compile(u'[\u4e00-\u9fa5]').search(line.split(',')[1])]
  img_name_list = [os.path.join(FLAGS.input_image_dir, img_label[0]) for img_label in img_label_list]
  label_list = [img_label[1].strip() for img_label in img_label_list]
  
  # input_image_str = [open(img_name, 'rb').read() for img_name in img_name_list]

  session_config = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.2  #占用30%显存
  with tf.Session(config=session_config) as sess:
    sess.run([
      tf.global_variables_initializer(),
      tf.local_variables_initializer(),
      tf.tables_initializer()])
    saver.restore(sess, checkpoint)
    count=0
    count_f=0
    error_sample_txt = os.path.join(FLAGS.exp_dir, 'log/model_{}_error.txt'.format(check_num))

    for i in range(len(img_name_list)):
      
      img = Image.open(img_name_list[i]).convert('L') # 将合成RGB彩色图像统一为灰度图像
      if FLAGS.padding:
        img = padding_image(img)
      img_buff = io.BytesIO()
      img.save(img_buff, format='jpeg')
      word_crop_jpeg = img_buff.getvalue()
      
      sess_outputs = sess.run(fetches, feed_dict={input_image_str_tensor: word_crop_jpeg})#input_image_str[i]})
      
      # print(sess.run(predictions_dict['Forward/labels'], feed_dict={input_image_str_tensor: input_image_str[i]}))
      # print(sess.run(recognitions, feed_dict={input_image_str_tensor: input_image_str[i]}))
      
      # print('Recognized text: {}'.format(sess_outputs['recognition_text'].decode('utf-8')))
      # fw=open('Chinese_aster/data/result_txt','a')
      # fw.write(img_name[i])
      # fw.write('{}'.format(sess_outputs['recognition_text'].decode('utf-8'))+'\n')
      # rectified_image = sess_outputs['rectified_images'][0]
      # rectified_image_pil = Image.fromarray((128 * (rectified_image + 1.0)).astype(np.uint8))
      # input_image_dir = 'Chinese_aster/result/'
      # rectified_image_save_path = os.path.join(input_image_dir, img_name[i])
      # rectified_image_pil.save(rectified_image_save_path)
      # print('Rectified image saved to {}'.format(rectified_image_save_path))
      # print(sess_outputs['recognitions'])

      # max_index = 0
      # max_score = float("inf")
      # for i in range(len(sess_outputs['recognitions_score'])):
      #   if sess_outputs['recognitions_score'][i] <= max_score:
      #     max_index = i

      predict_text_list = sess_outputs['recognitions'][0][0]
      predict_string = ""
      for char in predict_text_list:
        char = char.decode('utf-8')
        predict_string += char
      # print(predict_string, label_list[i])


      # predict=sess_outputs['recognition_text'].decode('utf-8')
      predict_string = predict_string.replace('（', '(').replace('）',')')
      label = label_list[i].replace(' ', '').replace('"', '').replace('（', '(').replace('）',')')
      count_f+=1
      
      # print(label,predict_string)
      if predict_string==label:
        count+=1
      else:
        with open(error_sample_txt, 'a') as fw:
          fw.write(img_name_list[i].split('/')[-1])
          fw.write('\t')
          fw.write(label)
          fw.write('\t')
          fw.write(predict_string)
          fw.write('\n')
      print(count_f)
    print(count/count_f)






if __name__ == '__main__':
  tf.app.run()
