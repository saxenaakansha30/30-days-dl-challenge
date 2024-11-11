# Problem: Implement YOLO for object detection (tutorial-based approach to simplify)
# Yolov3 Config file: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# Yolov3 Weights File: https://pjreddie.com/media/files/yolov3.weights
# Yolov3 coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names


import cv2
import numpy as np

# Load YOLOv3 weights, configuration and COCO class names
weight_path = 'yolov3/yolov3.weights'
config_path = 'yolov3/yolov3.cfg'
names_path = 'yolov3/coco.names'

# Load the COCO class names
with open(names_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Load the YOLOV3 model
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)

# Set backend and target device
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the input image
image = cv2.imread("sample_images/dog_cat_human.webp")
(height, width) = image.shape[:2]

# Create Blob from input image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(608, 608), swapRB=True, crop=False)
net.setInput(blob)

# Get output layers name
layer_names = net.getLayerNames()
# print(layers_name)

# print(net.getUnconnectedOutLayers())
output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# print(output_layer_names) # ['yolo_82', 'yolo_94', 'yolo_106']

# Perform forward pass
layer_outputs = net.forward(output_layer_names)

# Initialize List for Detection Results
boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:] # Gives the class scores (after the 5 bounding box parameters).
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            box = detection[:4] * np.array([width, height, width, height])
            (centerX, centerY, box_width, box_height) = box.astype("int")

            # Get the top left co-ordinates
            x = int(centerX - (box_width / 2))
            y = int(centerY - (box_height / 2))

            # Save the detections
            boxes.append([x, y, int(box_width), int(box_height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Use Non-Maxima Suppression to remove overlapping boundary boxes
supressed_indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw boundary boxes and labels.
for index in supressed_indices:
    (x, y) = (boxes[index][0], boxes[index][1])
    (w, h) = (boxes[index][2], boxes[index][3])

    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{class_names[class_ids[index]]}: {confidences[index]:.2f}"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_ITALIC, 0.5, color, 2)

# Display the final image
cv2.imshow("YOLO Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()