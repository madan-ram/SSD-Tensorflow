import os, sys
import glob
from random import choice
from datasets import pascalvoc_to_tfrecords
from notebooks.visualization import plt_bboxes
import cv2
import numpy as np

if __name__ == '__main__':
	dataset_dir = sys.argv[1]

	annotation_fpath_lst = glob.glob(os.path.join(dataset_dir, 'Annotations', "*.xml"))
	image_dir = os.path.join(dataset_dir, 'JPEGImages')

	annotation_fpath = None
	file_id = None
	while True:
		annotation_fpath = choice(annotation_fpath_lst)
		file_id = annotation_fpath.split('/')[-1].split('.')[0]
		image_path = os.path.join(image_dir, file_id+'.jpg')

		if os.path.isfile(image_path):
			print(image_path)
			break

	image_data, shape, bboxes, labels, labels_text, difficult, truncated = pascalvoc_to_tfrecords._process_image(dataset_dir, file_id)
	
	nparr = np.fromstring(image_data, np.uint8)
	image_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

	image_data = cv2.normalize(image_data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	print(labels)
	plt_bboxes(image_data, np.asarray(labels), [1.0 for _ in labels], np.asarray(bboxes), class_id_to_class_text={}, figsize=(10,10), linewidth=1.5)