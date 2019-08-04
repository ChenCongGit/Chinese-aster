# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Chinese_aster/protos/eval.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Chinese_aster.protos.preprocessor_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Chinese_aster/protos/eval.proto',
  package='Chinese_aster.protos',
  serialized_pb=_b('\n\x1f\x43hinese_aster/protos/eval.proto\x12\x14\x43hinese_aster.protos\x1a\'Chinese_aster/protos/preprocessor.proto\"\x97\x04\n\nEvalConfig\x12\x1e\n\x12num_visualizations\x18\x01 \x01(\r:\x02\x31\x30\x12\'\n\x18only_visualize_incorrect\x18\x0e \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x0cnum_examples\x18\x02 \x01(\r:\x04\x35\x30\x30\x30\x12\x1f\n\x12\x65val_interval_secs\x18\x03 \x01(\r:\x03\x33\x30\x30\x12\x14\n\tmax_evals\x18\x04 \x01(\r:\x01\x30\x12\x19\n\nsave_graph\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\"\n\x18visualization_export_dir\x18\x06 \x01(\t:\x00\x12\x15\n\x0b\x65val_master\x18\x07 \x01(\t:\x00\x12(\n\x0bmetrics_set\x18\x08 \x01(\t:\x13recognition_metrics\x12\x15\n\x0b\x65xport_path\x18\t \x01(\t:\x00\x12!\n\x12ignore_groundtruth\x18\n \x01(\x08:\x05\x66\x61lse\x12\"\n\x13use_moving_averages\x18\x0b \x01(\x08:\x05\x66\x61lse\x12\"\n\x13\x65val_instance_masks\x18\x0c \x01(\x08:\x05\x66\x61lse\x12 \n\x11\x65val_with_lexicon\x18\x0f \x01(\x08:\x05\x66\x61lse\x12I\n\x18\x64\x61ta_preprocessing_steps\x18\r \x03(\x0b\x32\'.Chinese_aster.protos.PreprocessingStep')
  ,
  dependencies=[Chinese_aster.protos.preprocessor_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_EVALCONFIG = _descriptor.Descriptor(
  name='EvalConfig',
  full_name='Chinese_aster.protos.EvalConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_visualizations', full_name='Chinese_aster.protos.EvalConfig.num_visualizations', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='only_visualize_incorrect', full_name='Chinese_aster.protos.EvalConfig.only_visualize_incorrect', index=1,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_examples', full_name='Chinese_aster.protos.EvalConfig.num_examples', index=2,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=5000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_interval_secs', full_name='Chinese_aster.protos.EvalConfig.eval_interval_secs', index=3,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=300,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_evals', full_name='Chinese_aster.protos.EvalConfig.max_evals', index=4,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='save_graph', full_name='Chinese_aster.protos.EvalConfig.save_graph', index=5,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='visualization_export_dir', full_name='Chinese_aster.protos.EvalConfig.visualization_export_dir', index=6,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_master', full_name='Chinese_aster.protos.EvalConfig.eval_master', index=7,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='metrics_set', full_name='Chinese_aster.protos.EvalConfig.metrics_set', index=8,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("recognition_metrics").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='export_path', full_name='Chinese_aster.protos.EvalConfig.export_path', index=9,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ignore_groundtruth', full_name='Chinese_aster.protos.EvalConfig.ignore_groundtruth', index=10,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_moving_averages', full_name='Chinese_aster.protos.EvalConfig.use_moving_averages', index=11,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_instance_masks', full_name='Chinese_aster.protos.EvalConfig.eval_instance_masks', index=12,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_with_lexicon', full_name='Chinese_aster.protos.EvalConfig.eval_with_lexicon', index=13,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data_preprocessing_steps', full_name='Chinese_aster.protos.EvalConfig.data_preprocessing_steps', index=14,
      number=13, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=99,
  serialized_end=634,
)

_EVALCONFIG.fields_by_name['data_preprocessing_steps'].message_type = Chinese_aster.protos.preprocessor_pb2._PREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['EvalConfig'] = _EVALCONFIG

EvalConfig = _reflection.GeneratedProtocolMessageType('EvalConfig', (_message.Message,), dict(
  DESCRIPTOR = _EVALCONFIG,
  __module__ = 'Chinese_aster.protos.eval_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.EvalConfig)
  ))
_sym_db.RegisterMessage(EvalConfig)


# @@protoc_insertion_point(module_scope)