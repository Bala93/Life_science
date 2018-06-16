import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import json
import urllib.request as urllib
import cv2
import argparse
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width,im_height)  = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Infer images in a folder')
    
    parser.add_argument(
        '--model_file',
        required = True,
        type = str,
        help = 'provide model file' )

    parser.add_argument(
        '--input_url',
        required = True,
        type = str,
        help = 'provide url which gives the video stream')

    parser.add_argument(
        'label_file',
        required = True,
        type = str,
        help = 'provide label_file')


    # Settings
    opt = parser.parse_args()
    model_file  = opt.model_file
    input_url  = opt.input_url
    label_file  = opt.label_file 
    NUM_CLASSES = 1
    
    # Label settings
    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Initializing the graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_file,'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')

    # Load the images from video
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
    
            tensor_dict = {}
            detection_fields = ['num_detections','detection_boxes','detection_scores','detection_classes']#,'detection_masks']
            
            for key in detection_fields:
                tensor_name = key + ':0'
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            while (1):

                # Camera url input
                imgresp = urllib.urlopen(input_url)

                # Decode the input
                imgnp  = np.array(bytearray(imgresp.read()),dtype=np.uint8)
                img_cv2_bgr = cv2.imdecode(imgnp,-1)

                # Convert to RGB and make it as a square image
                image_np_whole = cv2.cvtColor(img_cv2_bgr,cv2.COLOR_BGR2RGB)
                height,width,_ = image_np_whole.shape
                diff_  = width - height
                scale_ = int(diff_ / 2)
                end_   = width - scale_;
                image_np_resized = image_np_whole[:,scale_:end_,:]

                # Obtain the predicted output 
                output_dict  = sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image_np_resized,0)})
                
                output_dict['num_detections']    = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes']   = output_dict['detection_boxes'][0]
                output_dict['detection_scores']  = output_dict['detection_scores'][0]


                # Visualize the bounding box
                _,metric_box,metric_scores = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np_whole,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    shift_right = scale_,
                    line_thickness=8,
                    use_normalized_coordinates=True,
                )
                
                # Showcase the result in a opencv video stream
                img_cv2_bgr = cv2.cvtColor(image_np_whole,cv2.COLOR_BGR2RGB)
                cv2.imshow('frame',img_cv2_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break