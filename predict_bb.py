import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nets import ssd_avt_vgg, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
from datasets.avt_2020_v1 import VOC_LABELS


slim = tf.contrib.slim

class_id_to_class_text = {}
for class_text, (cls_id, dtype) in VOC_LABELS.items():
  class_id_to_class_text[cls_id] = class_text
  
NUM_CLASSES=4
ssd_class = ssd_avt_vgg.SSDNet
ssd_params = ssd_class.default_params._replace(num_classes=NUM_CLASSES)
ssd_net = ssd_class(ssd_params)
ssd_shape = ssd_net.params.img_shape

# Input placeholder.
net_shape = ssd_net.params.img_shape

from preprocessing import tf_image
import config_info
DATA_FORMAT = config_info.DATA_FORMAT

img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
if DATA_FORMAT == 'NCHW':
  img_input = tf.transpose(img_input, perm=(2, 0, 1))

# Evaluation pre-processing: resize to SSD net shape.
# image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
#     img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

image_pre = tf.compat.v1.image.resize_image_with_pad(img_input, net_shape[0], net_shape[1],
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
image_4d = tf.expand_dims(image_pre, 0)

# Main image processing routine.
def process_image(isess, img, img_pre, num_classes, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300), bboxes=None):
    bbox_img = tf.constant([[0., 0., 1., 1.]])
    if bboxes is None:
        bboxes = bbox_img
    else:
        bboxes = tf.concat([bbox_img, bboxes], axis=0)
    # Split back bounding boxes.
    bbox_img = bboxes[0]


    image_4d = tf.expand_dims(img, 0)
    # Run SSD network.
    rimg, pimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, img_pre, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=num_classes, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    
    return rclasses, rscores, rbboxes, pimg


# Test on some demo image and visualize output.
# path = '/content/project/SSD-Tensorflow/datasets/invoice_data/JPEGImages/X51005719886resized.jpg'
# path = '/content/project/SSD-Tensorflow/datasets/invoice_data/JPEGImages/X51005719904resized.jpg'
path = '/content/project/SSD-Tensorflow/datasets/invoice_data/JPEGImages/X51005745296resized.jpg'
ckpt_filename = './logs/model.ckpt-300'

img = mpimg.imread(path)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
with slim.arg_scope(ssd_net.arg_scope(data_format=DATA_FORMAT)):
  predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=tf.AUTO_REUSE)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
with tf.Session(config=config) as isess:
    # Restore SSD model.
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)
    # resized = cv2.resize(img, net_shape, interpolation = cv2.INTER_AREA)
    rclasses, rscores, rbboxes, pimg =  process_image(isess, img, image_pre, NUM_CLASSES)
    pimg = np.asarray(pimg, dtype="uint8")

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    visualization.plt_bboxes(pimg, rclasses, rscores, rbboxes, class_id_to_class_text)
