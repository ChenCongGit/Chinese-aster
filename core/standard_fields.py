class InputDataFields(object):
  image = 'image'
  original_image = 'original_image'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_text = 'groundtruth_text'
  groundtruth_text_with_blank = 'groundtruth_text_with_blank'
  groundtruth_keypoints = 'groundtruth_keypoints'
  lexicon = 'lexicon'


class TfExampleFields(object):
  image_encoded = 'image/encoded'
  image_format = 'image/format'  # format is reserved keyword
  filename = 'image/filename'
  channels = 'image/channels'
  colorspace = 'image/colorspace'
  height = 'image/height'
  width = 'image/width'
  source_id = 'image/source_id'
  transcript_gt_with_blank = 'image/transcript_gt_with_blank'
  transcript_gt = 'image/transcript_gt'
  lexicon = 'image/lexicon'
  keypoints = 'image/keypoints'
