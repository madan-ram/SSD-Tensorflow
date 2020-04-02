import os, sys
import glob
from random import choice
from datasets import pascalvoc_to_tfrecords
from notebooks.visualization import plt_bboxes
import cv2
import numpy as np
from datasets.pascalvoc_common import VOC_LABELS

# dataset_dir = '/content/project/SSD-Tensorflow/datasets/invoice_data/'
dataset_dir = '/Users/madanram/SoulOfCoder/SSD-Tensorflow/datasets/VOC2012/train/'


class_id_to_class_text = {}
for class_text, (cls_id, dtype) in VOC_LABELS.items():
  class_id_to_class_text[cls_id] = class_text
annotation_fpath_lst = glob.glob(os.path.join(dataset_dir, 'Annotations', "*.xml"))
image_dir = os.path.join(dataset_dir, 'JPEGImages')

with open('data.csv', 'w') as fw:
	fw.write(','.join(['img_path', 'img_h', 'img_w', 'y', 'x', 'h', 'w', 'label'])+'\n')
	for annotation_fpath in annotation_fpath_lst:
		file_id = annotation_fpath.split('/')[-1].split('.')[0]
		image_data, shape, bboxes, labels, labels_text, difficult, truncated = pascalvoc_to_tfrecords._process_image(dataset_dir, file_id)
		img_path = os.path.join(image_dir, file_id+'.jpg')
		for (y, x, h, w), label, _, _ in zip(bboxes, labels_text, difficult, truncated):
			img_h, img_w, _ = shape
			fw.write(','.join(map(str, [img_path, img_h, img_w, y, x, h, w, label]))+'\n')