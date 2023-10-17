from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
# additional lib to run added methods below
import cv2
import numpy as np
from utils import *
MODEL_PATH = 'model/retina.pth'
#MODEL_PATH = 'model/rcnn.pth'

class Detectron_Model:
    def __init__(self, model_path = MODEL_PATH, **kwargs):
        self.cfg = self.create_model_config(model_path = model_path, **kwargs)
        self.model = self.get_model(model_config = self.cfg)
    
    def create_model_config(self, model_path, treshold = 0.85, num_classes = 2 ,model_type="retinanet", **kwargs):
        # detectron2 model configuration
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        if model_type == "retinanet":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
            
            print('1')
        elif model_type == "faster_rcnn":
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            print('2')

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_TRESH_TEST = treshold
        
        
        return cfg

    def get_model(self, model_config):
        predictor = DefaultPredictor(model_config)
        return predictor
    
    def predict(self, image):
        prediction = self.model(image)
        return prediction
    
def NMS(boxes, overlapThresh = 0.4):
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > 0.87:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

#  modules needed to plot prediction
def get_bbox_data(prediction):
    response = []
    boxes = prediction['instances'].pred_boxes.tensor
    for entry in boxes:
        entry = entry.detach().to('cpu').numpy()
        x0,y0,x1,y1 = entry
        response.append((x0,y0,x1,y1))
    return response

def compute_size ( focal_lenght, dimx, x_dim, y_dim):
    W = dimx / 23.4
    s1 = x_dim
    s2 = y_dim
    L = int(focal_lenght)
    S1 = (s1 * W) / L
    S2 = (s2 * W) / L
    S = S1 * S2
    return S / 1000


def plot_prediction(image, outputs, focal_lenght, color = (0, 255, 0), thickness = 4):
  bbox_data = get_bbox_data(outputs)
  bbox_data = np.array(bbox_data)
  bbox_data = NMS(bbox_data)
  dim = np.shape(image) 
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 2

  for entry in bbox_data:
    x_dim= abs(int(entry[0]) - int(entry[2]))
    y_dim= abs(int(entry[1]) - int(entry[3]))
    whole_area = dim[1] * dim [2]
    bbox_surface = x_dim * y_dim
    if (focal_lenght == None):
        size_ratio= (bbox_surface/whole_area)
        size_ratio=round(size_ratio,3)
        image = cv2.putText(image, str(size_ratio), ( int(entry[0]),int(entry[1] - 10)  ), font, fontScale, color, 4, cv2.LINE_AA)
    else:
        S = compute_size(focal_lenght, dim[1], x_dim, y_dim)
        S=round(S,3)
       
        image = cv2.putText(image, str(S), ( int(entry[0]),int(entry[1] - 10)  ), font, fontScale, color, 4, cv2.LINE_AA)

    image = cv2.rectangle(image, (int(entry[0]), int(entry[1])), (int(entry[2]), int(entry[3])), color = color, thickness = thickness)
    
    
  return image