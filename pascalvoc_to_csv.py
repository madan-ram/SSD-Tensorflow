import os, sys
import glob
from random import choice
from datasets import pascalvoc_to_tfrecords
from notebooks.visualization import plt_bboxes
import cv2
import numpy as np
from datasets.pascalvoc_common import VOC_LABELS
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
import tensorflow as tf
from notebooks.visualization import plt_bboxes
import matplotlib.pyplot as plt

input_height, input_width = 1024, 1024

dataset_dir = '/content/project/SSD-Tensorflow/datasets/invoice_data/'
# dataset_dir = '/Users/madanram/SoulOfCoder/SSD-Tensorflow/datasets/VOC2012/train/'


class_id_to_class_text = {}
for class_text, (cls_id, dtype) in VOC_LABELS.items():
  class_id_to_class_text[cls_id] = class_text
annotation_fpath_lst = glob.glob(os.path.join(dataset_dir, 'Annotations', "*.xml"))
image_dir = os.path.join(dataset_dir, 'JPEGImages')

with open('data.csv', 'w') as fw:
    fw.write(','.join(['img_path', 'img_h', 'img_w', 'y', 'x', 'h', 'w', 'label'])+'\n')
    for annotation_fpath in annotation_fpath_lst:
        file_id = annotation_fpath.split('/')[-1].split('.')[0]
        image_data, shape, bboxes, labels, labels_text, difficult, truncated = pascalvoc_to_tfrecords._process_image(dataset_dir, file_id, label_id_map=VOC_LABELS)
        img_path = os.path.join(image_dir, file_id+'.jpg')
        img_t = tf.constant(cv2.imread(img_path))
        labels_t = tf.constant(labels)
        bboxes_t = tf.constant(bboxes)
        resized_t, labels_t, bboxes_t, bbox_img_t = preprocess_for_eval(img_t, labels, bboxes_t)
        # resized_t, labels_t, bboxes_t = resize_image_bboxes_with_crop_or_pad(img_t, bboxes_t, input_height, input_width)
        

        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            resized_img, resized_bboxes, _ = sess.run([resized_t, bboxes_t, bbox_img_t])
            # print(labels, resized_bboxes)
            for (y, x, h, w), label, _, _ in zip(resized_bboxes, labels_text, difficult, truncated):
                img_h, img_w = input_height, input_width

                fw.write(','.join(map(str, [img_path, img_h, img_w, y, x, h, w, label]))+'\n')