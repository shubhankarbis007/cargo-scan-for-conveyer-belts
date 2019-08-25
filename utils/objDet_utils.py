# from google.cloud import vision
# from google.cloud.vision import types
import os
import time
import sys
import cv2
sys.path.append('..')
from utils.app_utils import *
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import io
#from google.cloud import vision
#from google.cloud.vision import types
#client = vision.ImageAnnotatorClient()


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/frosted007/Documents/Cargo_Scan/model/cargo.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/frosted007/Documents/Cargo_Scan/model/cargo.pbtxt'

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

i_image=-1

def detect_objects(image_np, sess, detection_graph, crop_q):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualize coordinates of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    coordinates = vis_util.return_coordinates(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8,
                            min_score_thresh=0.20)  
    
    if (coordinates):
        global i_image
        i_image=i_image+1
        xmin = coordinates[0][2] 
        ymin = coordinates[0][0]
        xmax = coordinates[0][3]
        ymax = coordinates[0][1]
        if(True):
            cropped = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
            print("type of image is ", type(cropped))
            # cv2.imshow("cropped", cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print('cropped')

            crop_q.put(cropped)
            '''
            #write to file if file does not exist
            # path1 = "/home/mohak/Desktop/image/c1.jpg"
            # path2 = "/home/mohak/Desktop/image/c2.jpg"
            # path3 = "/home/mohak/Desktop/image/c3.jpg"
            path = "/home/mohak/Desktop/image/img"+str(i_image)+".jpg"

            if(not os.path.exists(path)):
                cv2.imwrite(path, cropped)
            # elif(not os.path.exists(path2)):
            #     cv2.imwrite('/home/mohak/Desktop/image/c2.jpg', cropped)
            # elif(not os.path.exists(path1)):
            #     cv2.imwrite('/home/mohak/Desktop/image/c1.jpg', cropped)
            else:
                pass

            #cv2.imwrite('/home/mohak/Desktop/image/cropped.jpg', cropped) 
            # cv2.imwrite('/home/mohak/Desktop/image/cropped'+str(i_image)+'.jpg', cropped)
            # print("size of queue is", L.qsize())
            '''
    return image_np


def worker(input_q, output_q, crop_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        print("inside worker!")
        fps.update()
        frame = input_q.get()

        # Check frame object is a 2-D array (video) or 1-D (webcam)
        if len(frame) == 2:
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            #Thread(target=output_q.put, args=(frame[0], detect_objects(frame_rgb, sess, detection_graph)),daemon=True).start()
            output_q.put((frame[0], detect_objects(frame_rgb, sess, detection_graph, crop_q)))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Thread(target=output_q.put, args=(detect_objects(frame_rgb, sess, detection_graph)),daemon=True).start()
            output_q.put(detect_objects(frame_rgb, sess, detection_graph, crop_q))
    fps.stop()
    sess.close()



def crop_worker(crop_q):
    i = 0
    while True:
        frame = crop_q.get()
        print("size of the crop queue is ", crop_q.qsize())
        cv2.imwrite("cropped_Image"+str(i)+".png", frame)
        print("the text is .....")
        #cv2.imshow("cropped image", frame)
        #your OCR function goes here 

        #write to file if file does not exist
        #path1 = "/home/mohak/Desktop/image/c1.jpg"
        # path2 = "/home/mohak/Desktop/image/c2.jpg"
        # path3 = "/home/mohak/Desktop/image/c3.jpg"
        #path = "/home/mohak/Desktop/image/img"+str(i_image)+".jpg"
        #if(not os.path.exists(path1)):
        #    cv2.imwrite(path1, frame)
        # elif(not os.path.exists(path2)):
        #     cv2.imwrite('/home/mohak/Desktop/image/c2.jpg', cropped)
        # elif(not os.path.exists(path1)):
        #     cv2.imwrite('/home/mohak/Desktop/image/c1.jpg', cropped)
        #else:
        #    pass

        #with io.open(path1, 'rb') as image_file:
        #    content = image_file.read()
        
        #frame = cv2.imencode('.jpg', frame)[1].tostring()
        #image = vision.types.Image(content=frame)
        #response = client.text_detection(image=image)
        #texts = response.text_annotations
        #print("GOT RESPONSE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #for text in texts:
        #    print("description",text.description)
            # File_object = open(r"/home/mohak/Desktop/image/text.txt","a+")
            # File_object.write(text.description)
            # File_object.close()
