import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load TF SavedModel detection function
detect_fn = tf.saved_model.load("exported-model/saved_model")

# Load label map
category_index = label_map_util.create_category_index_from_labelmap("annotations/label_map.pbtxt", use_display_name=True)

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, image_np = cap.read()
    if not ret:
        break

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0))

    # Run detection
    detections = detect_fn(input_tensor)

    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=0.68,
        agnostic_mode=False)

    # Show image with detections
    cv2.imshow('Object Detection Webcam', image_np)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
