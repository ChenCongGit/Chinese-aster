import os
import logging
from scipy import misc

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook


class ProfileAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, at_step=None, checkpoint_dir=None, trace_level=tf.RunOptions.FULL_TRACE):
    self._at_step = at_step
    self._do_profile = False
    self._writer = tf.summary.FileWriter(checkpoint_dir)
    self._trace_level = trace_level

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfileAtStepHook.")
  
  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._do_profile:
      options = tf.RunOptions(trace_level=self._trace_level)
    else:
      options = None

    graph = tf.get_default_graph()

    # 添加rotated_image张量运算和预处理图像
    rotated_image_tensor = graph.get_tensor_by_name(name="Input/RandomRotate/PyFunc:0")
    image_to_float_tensor = graph.get_tensor_by_name(name="Input/RandomRotate/ToFloat:0")
    resized_image_tensor = graph.get_tensor_by_name(name="Input/ResizeRandomMethod/Merge:0")
    preprocessed_image_tensor = graph.get_tensor_by_name(name='Input/RandomAdjustBrightness/clip_by_value:0')

    # 添加prediction_dict张量运算
    recognitions_1 = graph.get_tensor_by_name(name='clone_0/Postprocess/Forward/Postprocess/hash_table_1_Lookup:0')
    recognitions_2 = graph.get_tensor_by_name(name='clone_0/Postprocess/Backward/Postprocess/hash_table_3_Lookup:0')

    # groundtruth张量
    # groundtruth_tensor = graph.get_tensor_by_name(name='Input/Reshape_7:0')

    return tf.train.SessionRunArgs([self._global_step_tensor, rotated_image_tensor, image_to_float_tensor, resized_image_tensor, preprocessed_image_tensor, \
      [recognitions_1, recognitions_2]], options=options)

  def after_run(self, run_context, run_values):
    global_step = run_values.results[0] - 1
    if self._do_profile:
      self._do_profile = False
      self._writer.add_run_metadata(run_values.run_metadata,
                                    'trace_{}'.format(global_step), global_step)
      timeline_object = timeline.Timeline(run_values.run_metadata.step_stats)
      chrome_trace = timeline_object.generate_chrome_trace_format()
      chrome_trace_save_path = 'timeline_{}.json'.format(global_step)
      with open(chrome_trace_save_path, 'w') as f:
        f.write(chrome_trace)
      logging.info('Profile trace saved to {}'.format(chrome_trace_save_path))
    if global_step == self._at_step:
      self._do_profile = True

    # 保存预处理后图像
    rotated_image = run_values.results[1]
    # image_to_float = run_values.results[2]
    # resized_image = run_values.results[3]
    preprocessed_image = run_values.results[4]
    if global_step % 200 == 0:
      misc.imsave('./Chinese_aster/experiments/rotated_image/rotated_image_{}.jpg'.format(global_step), rotated_image)
      # misc.imsave('./Chinese_aster/experiments/image_to_float/image_to_float_{}.jpg'.format(global_step), image_to_float)
      # misc.imsave('./Chinese_aster/experiments/resized_image/resized_image_{}.jpg'.format(global_step), resized_image)
      misc.imsave('./Chinese_aster/experiments/preprocessed_image/preprocessed_image_{}.jpg'.format(global_step), preprocessed_image)

    # 保存训练预测结果
    predictions = run_values.results[5][0]
    if global_step % 200 == 0:
      for prediction in predictions:
        predict_string = ""
        for char in prediction:
          char = char.decode('utf-8')
          predict_string += char
        print(predict_string)
