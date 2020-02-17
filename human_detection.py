from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import argparse

def get_predictions(img_path, threshold):
    """
    A function to get the Human Detections of Pytorch Model on given image
    
    Parameters:
      - img_path : path of the input image
      - threshold : threshold value for prediction score
    Returns:
        Bounding boxes,class names and prediction scores of those detections which cross the threshold limit. 

    """
    # Loading the image into model as tensor and getting the model's predictions.
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]  # Mapping the labels to class names.           
    pred_boxes = [[(box[0], box[1]), (box[2], box[3])] for box in list(pred[0]['boxes'].detach().numpy())]        
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]  # Filtering prediction scores above threshold value. 
    
    # Selecting the predictions which cross the threshold value.
    pred_scores= pred_score[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return pred_boxes, pred_class, pred_scores


def draw_detections(img_path, threshold=0.6, rect_thickness=2, text_size=1, text_thickness=2):
    """
    A function that fetches the model's prediction and plots the Humans Detected in an image
    and saves the output thus created in the disk.
    It displays the total number of humans found,draws Red Bounding Boxes around the 
    predicted human detections with their probability scores just above the corresponding boxes. 

    Parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score, default value:0.6
      - rect_thickness - thickness of bounding box, default value:2
      - text_size - size of the probabilty scores' text, default value:1
      - text_thickness - thickness of the text, default value:2
    """
    # fetching the model's predictions
    boxes, pred_class, scores = get_predictions(img_path, threshold)
    img = cv2.imread(img_path)
    
    # Plotting the bounding boxes and respective probabilty scores while counting the number of human detections
    counter = 0
    for i in range(len(boxes)):
        if pred_class[i] == 'Person' :
            counter += 1 
            cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 0, 255), thickness=rect_thickness)
            cv2.putText(img,str(np.around(scores[i],3)), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_thickness)
    
    cv2.putText(img,"No. of Humans Predicted:"+str(counter),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),thickness = 2)
    
    print("The output is being saved in present directory as 'detections.jpg'")
    cv2.imwrite('detections.jpg'.format(threshold),img)

if __name__ == '__main__':

    # constructing the argument parser and parsing the arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-img_path", required=True,
        help="path to the image to be tested for human detection, madatory argument")
    
    ap.add_argument("-thresh", required=False, type=float, default=0.6,
        help="threshold value between 0 & 1 to control the prediction scores, optional argument, expects float value, default:0.6")

    ap.add_argument("-rect_th", required=False, type=int, default=2,
        help="thickness of rectangular bounding box, optional argument, expects integer value only, default:2")
    
    ap.add_argument("-text_sz", required=False, type=float, default=1.0,
        help="size of the probabilty scores' text over bounding box, optional argument, expects float value, default:1.0")
    
    ap.add_argument("-text_th", required=False, type=int, default=2,
        help="thickness of the text of bounding box, optional argument, expects integer value only, default:2")
    
    args = vars(ap.parse_args())

    # Fetching the arguments 
    img_path = args["img_path"]
    threshold = args["thresh"]
    rect_thickness = args["rect_th"]
    text_size = args["text_sz"]
    text_thickness = args["text_th"]
    
    # Get the pretrained object detection model and set the model for inference.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()


    # Categories from official PyTorch documentation for which the model was trained on.   
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'Person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Fetching model's predictions & Plotting the detections
    draw_detections(img_path, threshold, rect_thickness, text_size, text_thickness)

