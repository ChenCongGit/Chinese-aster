# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Chinese_aster/protos/convnet.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Chinese_aster.protos.hyperparams_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Chinese_aster/protos/convnet.proto',
  package='Chinese_aster.protos',
  serialized_pb=_b('\n\"Chinese_aster/protos/convnet.proto\x12\x14\x43hinese_aster.protos\x1a&Chinese_aster/protos/hyperparams.proto\"\xb6\x01\n\x07\x43onvnet\x12\x31\n\x08\x63rnn_net\x18\x01 \x01(\x0b\x32\x1d.Chinese_aster.protos.CrnnNetH\x00\x12.\n\x06resnet\x18\x02 \x01(\x0b\x32\x1c.Chinese_aster.protos.ResNetH\x00\x12\x37\n\x0bstn_convnet\x18\x03 \x01(\x0b\x32 .Chinese_aster.protos.StnConvnetH\x00\x42\x0f\n\rconvnet_oneof\"\x8d\x02\n\x07\x43rnnNet\x12\x46\n\x08net_type\x18\x01 \x01(\x0e\x32%.Chinese_aster.protos.CrnnNet.NetType:\rSINGLE_BRANCH\x12;\n\x10\x63onv_hyperparams\x18\x02 \x01(\x0b\x32!.Chinese_aster.protos.Hyperparams\x12$\n\x15summarize_activations\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x04tiny\x18\x04 \x01(\x08:\x05\x66\x61lse\"B\n\x07NetType\x12\x11\n\rSINGLE_BRANCH\x10\x00\x12\x10\n\x0cTWO_BRANCHES\x10\x01\x12\x12\n\x0eTHREE_BRANCHES\x10\x02\"\xf5\x02\n\x06ResNet\x12\x45\n\x08net_type\x18\x01 \x01(\x0e\x32$.Chinese_aster.protos.ResNet.NetType:\rSINGLE_BRANCH\x12\x43\n\tnet_depth\x18\x02 \x01(\x0e\x32%.Chinese_aster.protos.ResNet.NetDepth:\tRESNET_50\x12;\n\x10\x63onv_hyperparams\x18\x03 \x01(\x0b\x32!.Chinese_aster.protos.Hyperparams\x12$\n\x15summarize_activations\x18\x04 \x01(\x08:\x05\x66\x61lse\"B\n\x07NetType\x12\x11\n\rSINGLE_BRANCH\x10\x00\x12\x10\n\x0cTWO_BRANCHES\x10\x01\x12\x12\n\x0eTHREE_BRANCHES\x10\x02\"8\n\x08NetDepth\x12\r\n\tRESNET_30\x10\x00\x12\r\n\tRESNET_50\x10\x01\x12\x0e\n\nRESNET_100\x10\x02\"\x84\x01\n\nStnConvnet\x12;\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32!.Chinese_aster.protos.Hyperparams\x12$\n\x15summarize_activations\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x04tiny\x18\x03 \x01(\x08:\x05\x66\x61lse')
  ,
  dependencies=[Chinese_aster.protos.hyperparams_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_CRNNNET_NETTYPE = _descriptor.EnumDescriptor(
  name='NetType',
  full_name='Chinese_aster.protos.CrnnNet.NetType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SINGLE_BRANCH', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWO_BRANCHES', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='THREE_BRANCHES', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=489,
  serialized_end=555,
)
_sym_db.RegisterEnumDescriptor(_CRNNNET_NETTYPE)

_RESNET_NETTYPE = _descriptor.EnumDescriptor(
  name='NetType',
  full_name='Chinese_aster.protos.ResNet.NetType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SINGLE_BRANCH', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWO_BRANCHES', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='THREE_BRANCHES', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=489,
  serialized_end=555,
)
_sym_db.RegisterEnumDescriptor(_RESNET_NETTYPE)

_RESNET_NETDEPTH = _descriptor.EnumDescriptor(
  name='NetDepth',
  full_name='Chinese_aster.protos.ResNet.NetDepth',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESNET_30', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESNET_50', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESNET_100', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=875,
  serialized_end=931,
)
_sym_db.RegisterEnumDescriptor(_RESNET_NETDEPTH)


_CONVNET = _descriptor.Descriptor(
  name='Convnet',
  full_name='Chinese_aster.protos.Convnet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='crnn_net', full_name='Chinese_aster.protos.Convnet.crnn_net', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resnet', full_name='Chinese_aster.protos.Convnet.resnet', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stn_convnet', full_name='Chinese_aster.protos.Convnet.stn_convnet', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
    _descriptor.OneofDescriptor(
      name='convnet_oneof', full_name='Chinese_aster.protos.Convnet.convnet_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=101,
  serialized_end=283,
)


_CRNNNET = _descriptor.Descriptor(
  name='CrnnNet',
  full_name='Chinese_aster.protos.CrnnNet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='net_type', full_name='Chinese_aster.protos.CrnnNet.net_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='Chinese_aster.protos.CrnnNet.conv_hyperparams', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='summarize_activations', full_name='Chinese_aster.protos.CrnnNet.summarize_activations', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tiny', full_name='Chinese_aster.protos.CrnnNet.tiny', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _CRNNNET_NETTYPE,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=286,
  serialized_end=555,
)


_RESNET = _descriptor.Descriptor(
  name='ResNet',
  full_name='Chinese_aster.protos.ResNet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='net_type', full_name='Chinese_aster.protos.ResNet.net_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='net_depth', full_name='Chinese_aster.protos.ResNet.net_depth', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='Chinese_aster.protos.ResNet.conv_hyperparams', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='summarize_activations', full_name='Chinese_aster.protos.ResNet.summarize_activations', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESNET_NETTYPE,
    _RESNET_NETDEPTH,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=558,
  serialized_end=931,
)


_STNCONVNET = _descriptor.Descriptor(
  name='StnConvnet',
  full_name='Chinese_aster.protos.StnConvnet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='Chinese_aster.protos.StnConvnet.conv_hyperparams', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='summarize_activations', full_name='Chinese_aster.protos.StnConvnet.summarize_activations', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tiny', full_name='Chinese_aster.protos.StnConvnet.tiny', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
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
  serialized_start=934,
  serialized_end=1066,
)

_CONVNET.fields_by_name['crnn_net'].message_type = _CRNNNET
_CONVNET.fields_by_name['resnet'].message_type = _RESNET
_CONVNET.fields_by_name['stn_convnet'].message_type = _STNCONVNET
_CONVNET.oneofs_by_name['convnet_oneof'].fields.append(
  _CONVNET.fields_by_name['crnn_net'])
_CONVNET.fields_by_name['crnn_net'].containing_oneof = _CONVNET.oneofs_by_name['convnet_oneof']
_CONVNET.oneofs_by_name['convnet_oneof'].fields.append(
  _CONVNET.fields_by_name['resnet'])
_CONVNET.fields_by_name['resnet'].containing_oneof = _CONVNET.oneofs_by_name['convnet_oneof']
_CONVNET.oneofs_by_name['convnet_oneof'].fields.append(
  _CONVNET.fields_by_name['stn_convnet'])
_CONVNET.fields_by_name['stn_convnet'].containing_oneof = _CONVNET.oneofs_by_name['convnet_oneof']
_CRNNNET.fields_by_name['net_type'].enum_type = _CRNNNET_NETTYPE
_CRNNNET.fields_by_name['conv_hyperparams'].message_type = Chinese_aster.protos.hyperparams_pb2._HYPERPARAMS
_CRNNNET_NETTYPE.containing_type = _CRNNNET
_RESNET.fields_by_name['net_type'].enum_type = _RESNET_NETTYPE
_RESNET.fields_by_name['net_depth'].enum_type = _RESNET_NETDEPTH
_RESNET.fields_by_name['conv_hyperparams'].message_type = Chinese_aster.protos.hyperparams_pb2._HYPERPARAMS
_RESNET_NETTYPE.containing_type = _RESNET
_RESNET_NETDEPTH.containing_type = _RESNET
_STNCONVNET.fields_by_name['conv_hyperparams'].message_type = Chinese_aster.protos.hyperparams_pb2._HYPERPARAMS
DESCRIPTOR.message_types_by_name['Convnet'] = _CONVNET
DESCRIPTOR.message_types_by_name['CrnnNet'] = _CRNNNET
DESCRIPTOR.message_types_by_name['ResNet'] = _RESNET
DESCRIPTOR.message_types_by_name['StnConvnet'] = _STNCONVNET

Convnet = _reflection.GeneratedProtocolMessageType('Convnet', (_message.Message,), dict(
  DESCRIPTOR = _CONVNET,
  __module__ = 'Chinese_aster.protos.convnet_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.Convnet)
  ))
_sym_db.RegisterMessage(Convnet)

CrnnNet = _reflection.GeneratedProtocolMessageType('CrnnNet', (_message.Message,), dict(
  DESCRIPTOR = _CRNNNET,
  __module__ = 'Chinese_aster.protos.convnet_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.CrnnNet)
  ))
_sym_db.RegisterMessage(CrnnNet)

ResNet = _reflection.GeneratedProtocolMessageType('ResNet', (_message.Message,), dict(
  DESCRIPTOR = _RESNET,
  __module__ = 'Chinese_aster.protos.convnet_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.ResNet)
  ))
_sym_db.RegisterMessage(ResNet)

StnConvnet = _reflection.GeneratedProtocolMessageType('StnConvnet', (_message.Message,), dict(
  DESCRIPTOR = _STNCONVNET,
  __module__ = 'Chinese_aster.protos.convnet_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.StnConvnet)
  ))
_sym_db.RegisterMessage(StnConvnet)


# @@protoc_insertion_point(module_scope)
