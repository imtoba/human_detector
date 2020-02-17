# Human Detector
A pretrained Object Detection Model to detect Humans. 

## Requirements
The provided script has been found to be working with following required dependencies:
 * Python 3.6
 * PyTorch 1.4.0
 * torchvision 0.5.0
 * numpy 1.18.1
 * OpenCV 4.1.2


## Instructions
  ### Quick Run 
  * Download and open the provided folder 

  * To Reproduce the exact same outputs locally execute the command with the mandatory "img_path" argument :

    ```python3 human_detection.py  -img_path ./TopDownHumanDetection_4032x3024.jpg```

  * To test on some other images just change the path provided as:

    ```python3 human_detection.py  -img_path /path/to/test/image/```

### Other optional arguments
There are 4 other optional arguments to the provided script as well to tune the output image. They could be run as:

  ```python3 human_detection.py  -arg ARG_VALUE```

where 'arg' is to be replaced with one of the following arguments:
  
  * '-thresh': Threshold value between 0 & 1 to control the prediction scores, optional argument, expects float value, default:0.6
  
  * '-rect_th': Thickness of rectangular bounding box, optional argument, expects integer value only, default:2
  
  * '-text_sz': Text Size of the probabilty scores over bounding box, optional argument, expects float value, default:1.0
  
  * '-text_th': Thickness of the text of bounding box, optional argument, expects integer value only, default:2
## Net Average IoU Score:
  * The model predicted 24 humans out of which 20 were actually humans and 4 were not humans but human-shaped objects.(with threshold = 0.6)
  * There were 26 humans in the ground truth.Most of the humans that did't get detected were occluded/hidden due to other objects.
  * The net average IoU Score for the given image turned out to be **0.58** when tested with an apt threshold value of 0.6.
 

