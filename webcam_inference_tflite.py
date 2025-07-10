import cv2
import numpy as np
import tensorflow as tf
import re

# Load label map dari file .pbtxt
def load_labels(path):
    with open(path, 'r') as f:
        content = f.read()
    matches = re.findall(r'id:\s*(\d+)\s*name:\s*["\'](.+?)["\']', content)
    labels = {}
    for id_str, name in matches:
        labels[int(id_str)] = name
    return labels

labels = load_labels("annotations/label_map.pbtxt")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_flex.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Output details:")
for i, detail in enumerate(output_details):
    print(f"{i}: name={detail['name']}, shape={detail['shape']}")

# Mapping indeks output sesuai urutan dari model
scores_idx = 0     # [1, 10]
boxes_idx = 1      # [1, 10, 4]
count_idx = 2      # [1]
classes_idx = 3    # [1, 10]

# Set up webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam tidak bisa dibuka.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_shape = input_details[0]['shape']
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ambil hasil deteksi
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[count_idx]['index'])[0])
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]

    print(f"Jumlah deteksi: {num_detections}")

    h, w, _ = frame.shape
    for i in range(num_detections):
        score = scores[i]
        if score > 0.65:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (
                int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            )
            class_id = int(classes[i])
            class_name = labels.get(class_id, "unknown")
            label = f"{class_name} ({score:.2f})"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow('TFLite Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
