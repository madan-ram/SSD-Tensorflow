# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Definition of 512 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from tensorflow.python.ops import array_ops

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate

def multiply_list_items(myList): 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(input_tensor=per_entry_cross_ent, axis=-1)


def focal_loss_v2(prediction_tensor, target_labels, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, i_0, ..., i_{K-2},
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, i_0, ..., i_{K-2},
        1] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    target_label_shape = target_labels.get_shape()
    num_classes =  prediction_tensor.get_shape().as_list()[-1]
    eye_m = tf.eye(num_classes, dtype=prediction_tensor.dtype)
    target_tensor = tf.gather(eye_m, target_labels, batch_dims=0, name=None)
    
    t_shape = target_tensor.get_shape()
    target_tensor = tf.reshape(target_tensor, (t_shape[0], multiply_list_items(t_shape[1:-1]), t_shape[-1]), name=None)

    t_shape = prediction_tensor.get_shape()
    prediction_tensor = tf.reshape(prediction_tensor, (t_shape[0], multiply_list_items(t_shape[1:-1]), t_shape[-1]), name=None)
    
    loss = focal_loss(prediction_tensor, target_tensor, alpha=alpha, gamma=gamma)

    # Loss per label
    loss = tf.reshape(loss, target_label_shape)
    return loss

slim = tf.contrib.slim
# ssd_net.default_image_size = 1024

# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(input=x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
                
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
   """Construct a multibox layer, return a class and localization predictions.
   """
   net = inputs
   if normalization > 0:
     net = custom_layers.l2_normalization(net, scaling=True)
   # Number of anchors.
   num_anchors = len(sizes) + len(ratios)

   # Location.
   num_loc_pred = num_anchors * 4
   loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                        scope='conv_loc')
   loc_pred = custom_layers.channel_to_last(loc_pred)
   loc_pred = tf.reshape(loc_pred,
                       tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
   # Class prediction.
   num_cls_pred = num_anchors * num_classes
   cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                        scope='conv_cls')
   cls_pred = custom_layers.channel_to_last(cls_pred)
   cls_pred = tf.reshape(cls_pred,
                       tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
   return cls_pred, loc_pred

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 512 network.

    The default features layers with 512x512 image input are:
      conv4 ==> 64 x 64
      conv7 ==> 32 x 32
      conv8 ==> 16 x 16
      conv9 ==> 8 x 8
      conv10 ==> 4 x 4
      conv11 ==> 2 x 2
      conv12 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    # Anchor box square sizes
    anchor_size_bounds = [0.50, 0.50]
    anchor_steps = [16, 32, 64, 128, 256, 512, 1024]
    avg_anchor_size_min = np.asarray(anchor_steps)*anchor_size_bounds[0]
    avg_anchor_size_max = np.asarray(anchor_steps)*anchor_size_bounds[1]
    print(avg_anchor_size_min, avg_anchor_size_max, '=========================')
    anchor_sizes = list(zip(avg_anchor_size_min, avg_anchor_size_max))

    default_params = SSDParams(
        feat_layers=['block3_fused_block4', 'block4_fused_block5', 'block5_fused_block6', 'block6', 'block7', 'block8', 'block9'],
        img_shape=(1024, 1024),
        num_classes=5,
        no_annotation_label=5,
        feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        anchor_size_bounds=anchor_size_bounds,
        anchor_sizes=anchor_sizes,
        # Using Kmean clustering find best average aspect ration (centroid)
        anchor_ratios=[[1.5, 1.7, 2.2],
                       [1.5, 1.7, 2.2],
                       [1.7, 2.2, 2.7, 3.6, 4.7],
                       [1.6, 2.5, 3.3, 4, 5.0],
                       [1.6, 3.9, 7.7],
                       [1.5, 3.7, 7.0, 12.4],
                       [4.2, 6.3, 9.6]],
        anchor_steps=anchor_steps,
        anchor_offset=0.5,
        normalizations=[-1, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if params is not None:
          print("Setting params provided in class")
          self.params = params
        # if isinstance(params, SSDParams):
        #     self.params = params
        # else:
        #     self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_avt_vgg_deep'):
        """Network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!) TODO: look into this somthing, somthing phishe
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def layer_shape(layer):
    """Returns the dimensions of a 4D layer tensor.
    Args:
      layer: A 4-D Tensor of shape `[height, width, channels]`.
    Returns:
      Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if layer.get_shape().is_fully_defined():
        return layer.get_shape().as_list()
    else:
        static_shape = layer.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(input=layer), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(512, 512)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def fuse_conv_layer(fuse_to, fuse_from, num_filters, filter_size=2, strides=2):
  # Upsample and normalize
  fuse_from_num_of_filters = fuse_from.get_shape()[-1]
  kernel_size = (filter_size, filter_size)

  fuse_from = tf.keras.layers.Conv2DTranspose(fuse_from_num_of_filters, kernel_size, activation='relu', padding='SAME', strides=strides)(fuse_from)
  fuse_from_norm = custom_layers.l2_normalization(fuse_from, scaling=True)

  fuse_to_norm = custom_layers.l2_normalization(fuse_to, scaling=True)

  fused_layer = tf.concat([fuse_to_norm, fuse_from_norm], axis=3)
  
  fused_layer = Conv2D(num_filters, (1, 1), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='fusion_weight_layer')(fused_layer)
  # fused_layer = net = slim.conv2d(fused_layer, num_filters, [1, 1], stride=1, scope='fusion_weight_layer', padding='SAME')
  
  return fused_layer

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 512.
# =========================================================================== #
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_avt_vgg_deep'):
    """SSD net definition.
    """

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.compat.v1.variable_scope(scope, 'ssd_avt_vgg_deep', [inputs], reuse=reuse):
        # Block1
        # Conv  32  2 0 3
        # Conv  32  1 0 3
        # Conv  32  1 0 3
        # Conv  64  1 0 3
        # Max 64  2 0 2
        block = 'block1'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(inputs)
          net = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        # Block2
        # Conv  128 1 0 3
        # Max 128 2 0 3
        block = 'block2'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        # Block3
        # Conv  128 1 0 3
        # Max 128 2 0 2
        block = 'block3'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        # Block4
        # Conv  128 1 0 3
        # Conv  128 1 0 3
        # Conv  256 1 0 3
        # Conv  256 1 0 3
        # Max 256 2 0 2
        block = 'block4'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        block = 'block3_fused_block4'
        with tf.compat.v1.variable_scope(block):
          fuse_net = fuse_conv_layer(end_points['block3'], end_points['block4'], 256)
          end_points[block] = fuse_net

        # Block5
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Max 384 2 0 2
        block = 'block5'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net


        block = 'block4_fused_block5'
        with tf.compat.v1.variable_scope(block):
          fuse_net = fuse_conv_layer(end_points['block4'], end_points['block5'], 256)
          end_points[block] = fuse_net

        # Block6
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Max 384 2 0 2
        block = 'block6'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        block = 'block5_fused_block6'
        with tf.compat.v1.variable_scope(block):
          fuse_net = fuse_conv_layer(end_points['block5'], end_points['block6'], 256)
          end_points[block] = fuse_net

        # Block7
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Max 384 2 0 2
        block = 'block7'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net


        # Block7
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Max 384 2 0 2
        block = 'block8'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net

        # Block7
        # Conv  384 1 0 3
        # Conv  384 1 0 3
        # Max 384 2 0 2
        block = 'block9'
        with tf.compat.v1.variable_scope(block):
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_1')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_2')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_3')(net)
          net = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', kernel_initializer='he_normal', name='conv3x3_4')(net)
          net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(net)
          end_points[block] = net


        # # BLock6 TODO to include dilate
        # # Conv  32  6 0 3
        # # Additional SSD blocks.
        # # Block 6: let's dilate the hell out of it!
        # block = 'context_block'
        # with tf.variable_scope(block):
        #   net = slim.conv2d(net, 128, [3, 3], rate=6, scope='conv6')
        #   end_points[block] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        print(anchor_sizes)
        for i, layer in enumerate(feat_layers):
            print(layer,'shape is ->', end_points[layer].get_shape())
            with tf.compat.v1.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay)),
                        weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                        biases_initializer=tf.compat.v1.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.compat.v1.name_scope(scope, 'ssd_losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        for i in range(len(logits)):
            dtype = logits[i].dtype
            with tf.compat.v1.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(input_tensor=fpmask)

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.compat.v1.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(input=nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(input=nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(input_tensor=fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.compat.v1.name_scope('cross_entropy_pos'):
                    # print('logits ->', logits[i].get_shape())
                    # print('gclasses ->', gclasses[i].get_shape())
                    # print("=================================================")
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=gclasses[i])
                    # loss = focal_loss_v2(logits[i], gclasses[i])
                    loss = tf.compat.v1.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)

                with tf.compat.v1.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=no_classes)
                    # loss = focal_loss_v2(logits[i], no_classes)
                    loss = tf.compat.v1.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.compat.v1.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.compat.v1.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.compat.v1.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')

            # Add to EXTRA LOSSES TF.collection
            tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_loc)
