# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Chinese_aster/protos/rnn_cell.proto

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
  name='Chinese_aster/protos/rnn_cell.proto',
  package='Chinese_aster.protos',
  serialized_pb=_b('\n#Chinese_aster/protos/rnn_cell.proto\x12\x14\x43hinese_aster.protos\x1a&Chinese_aster/protos/hyperparams.proto\"\x83\x01\n\x07RnnCell\x12\x33\n\tlstm_cell\x18\x01 \x01(\x0b\x32\x1e.Chinese_aster.protos.LstmCellH\x00\x12\x31\n\x08gru_cell\x18\x02 \x01(\x0b\x32\x1d.Chinese_aster.protos.GruCellH\x00\x42\x10\n\x0ernn_cell_oneof\"\x90\x01\n\x08LstmCell\x12\x16\n\tnum_units\x18\x01 \x01(\r:\x03\x31\x32\x38\x12\x1c\n\ruse_peepholes\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x0b\x66orget_bias\x18\x03 \x01(\x02:\x01\x31\x12\x36\n\x0binitializer\x18\x04 \x01(\x0b\x32!.Chinese_aster.protos.Initializer\"Y\n\x07GruCell\x12\x16\n\tnum_units\x18\x01 \x01(\r:\x03\x31\x32\x38\x12\x36\n\x0binitializer\x18\x02 \x01(\x0b\x32!.Chinese_aster.protos.Initializer')
  ,
  dependencies=[Chinese_aster.protos.hyperparams_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_RNNCELL = _descriptor.Descriptor(
  name='RnnCell',
  full_name='Chinese_aster.protos.RnnCell',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lstm_cell', full_name='Chinese_aster.protos.RnnCell.lstm_cell', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gru_cell', full_name='Chinese_aster.protos.RnnCell.gru_cell', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
      name='rnn_cell_oneof', full_name='Chinese_aster.protos.RnnCell.rnn_cell_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=102,
  serialized_end=233,
)


_LSTMCELL = _descriptor.Descriptor(
  name='LstmCell',
  full_name='Chinese_aster.protos.LstmCell',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_units', full_name='Chinese_aster.protos.LstmCell.num_units', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=128,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_peepholes', full_name='Chinese_aster.protos.LstmCell.use_peepholes', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='forget_bias', full_name='Chinese_aster.protos.LstmCell.forget_bias', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initializer', full_name='Chinese_aster.protos.LstmCell.initializer', index=3,
      number=4, type=11, cpp_type=10, label=1,
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
  ],
  serialized_start=236,
  serialized_end=380,
)


_GRUCELL = _descriptor.Descriptor(
  name='GruCell',
  full_name='Chinese_aster.protos.GruCell',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_units', full_name='Chinese_aster.protos.GruCell.num_units', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=128,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initializer', full_name='Chinese_aster.protos.GruCell.initializer', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
  ],
  serialized_start=382,
  serialized_end=471,
)

_RNNCELL.fields_by_name['lstm_cell'].message_type = _LSTMCELL
_RNNCELL.fields_by_name['gru_cell'].message_type = _GRUCELL
_RNNCELL.oneofs_by_name['rnn_cell_oneof'].fields.append(
  _RNNCELL.fields_by_name['lstm_cell'])
_RNNCELL.fields_by_name['lstm_cell'].containing_oneof = _RNNCELL.oneofs_by_name['rnn_cell_oneof']
_RNNCELL.oneofs_by_name['rnn_cell_oneof'].fields.append(
  _RNNCELL.fields_by_name['gru_cell'])
_RNNCELL.fields_by_name['gru_cell'].containing_oneof = _RNNCELL.oneofs_by_name['rnn_cell_oneof']
_LSTMCELL.fields_by_name['initializer'].message_type = Chinese_aster.protos.hyperparams_pb2._INITIALIZER
_GRUCELL.fields_by_name['initializer'].message_type = Chinese_aster.protos.hyperparams_pb2._INITIALIZER
DESCRIPTOR.message_types_by_name['RnnCell'] = _RNNCELL
DESCRIPTOR.message_types_by_name['LstmCell'] = _LSTMCELL
DESCRIPTOR.message_types_by_name['GruCell'] = _GRUCELL

RnnCell = _reflection.GeneratedProtocolMessageType('RnnCell', (_message.Message,), dict(
  DESCRIPTOR = _RNNCELL,
  __module__ = 'Chinese_aster.protos.rnn_cell_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.RnnCell)
  ))
_sym_db.RegisterMessage(RnnCell)

LstmCell = _reflection.GeneratedProtocolMessageType('LstmCell', (_message.Message,), dict(
  DESCRIPTOR = _LSTMCELL,
  __module__ = 'Chinese_aster.protos.rnn_cell_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.LstmCell)
  ))
_sym_db.RegisterMessage(LstmCell)

GruCell = _reflection.GeneratedProtocolMessageType('GruCell', (_message.Message,), dict(
  DESCRIPTOR = _GRUCELL,
  __module__ = 'Chinese_aster.protos.rnn_cell_pb2'
  # @@protoc_insertion_point(class_scope:Chinese_aster.protos.GruCell)
  ))
_sym_db.RegisterMessage(GruCell)


# @@protoc_insertion_point(module_scope)