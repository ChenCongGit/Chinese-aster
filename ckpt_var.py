import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', "./Chinese_aster/experiments/demo/res_log_300k/model.ckpt-19510", "restore ckpt")
FLAGS = flags.FLAGS


def main(_):
    """
    查看tensorflow的模型文件model.ckpt保存的变量及变量的值
    """
    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('11111111111', "tensor_name: ", key, "tensor_shape: ", reader.get_tensor(key).shape)

    for var in slim.get_model_variables():
        print('22222222222', var)


if __name__ == "__main__":
    tf.app.run()