# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.avt_2020_v1 import VOC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def negative_to_zero(val):
    if val<0.0:
        return 0.0
    return val

def above_1_to_1(val):
    if val>1.0:
        return 1.0
    return val
    
def _process_image(directory, name, label_id_map=VOC_LABELS):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.compat.v1.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        # Skip unlabelled object
        if label == 'unlabelled' or label is None:
            attr_label = None
            for attr in obj.find('attributes').findall('attribute'):
                attr_label = attr.find('key').text
                break
            if attr_label is not None and attr_label != 'unlabelled':
                label = attr_label
            else:
                continue

        # HACK: Remove it for correct output
        # labels.append(label)
        # label = label+1
        try:
            labels.append(int(label_id_map[label][0]))
        except KeyError as ke:
            print('SKilling label {} in file_id {}'.format(label, name))

        labels_text.append(str(label).encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append(tuple(map(above_1_to_1, map(negative_to_zero, [float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ]))))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


# def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
#     """Loads data from image and annotations files and add them to a TFRecord.

#     Args:
#       dataset_dir: Dataset directory;
#       name: Image name to add to the TFRecord;
#       tfrecord_writer: The TFRecord writer to use for writing.
#     """
#     image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
#         _process_image(dataset_dir, name)
#     example = _convert_to_example(image_data, labels, labels_text,
#                                   bboxes, shape, difficult, truncated)
#     tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.io.gfile.exists(dataset_dir):
        tf.io.gfile.makedirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    count_num_exp = 0
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]

                image_data, shape, bboxes, labels, labels_text, difficult, truncated = _process_image(dataset_dir, img_name)
                example = _convert_to_example(image_data, labels, labels_text,
                                              bboxes, shape, difficult, truncated)
                tfrecord_writer.write(example.SerializeToString())
                count_num_exp += 1
                i += 1
                j += 1
            fidx += 1

    print("\nNumer of samples %d"%count_num_exp)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('Finished converting the Pascal VOC dataset!')
