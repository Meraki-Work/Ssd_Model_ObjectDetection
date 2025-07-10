from flask import Flask, Response
import cv2
import numpy as np
import tensorflow as tf
import time

# STEP 1 Load SavedModel SSD
detect_fn = tf.saved_model.load('exported-model/saved_model')

# STEP 2 Kelas deteksi (ubah sesuai label_map.pbtxt-mu)
LABELS = {1: 'cengkeh_matang', 2: 'cengkeh_mentah', 3: 'cengkeh_batang'}

# STEP 3 MJPEG Flask App
app = Flask(__name__)
cap = cv2.VideoCapture(0)

def detect_objects(frame):
    h, w, _ = frame.shape
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)

    input_shape = input_tensor.shape

    # Time profiling
    t0 = time.time()
    detections = detect_fn(input_tensor)
    t1 = time.time()

    inference_time_ms = (t1 - t0) * 1000

    # Post-processing
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    t2 = time.time()
    post_time_ms = (t2 - t1) * 1000
    total_time_ms = (t2 - t0) * 1000

    detected_labels = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top) = (int(xmin * w), int(ymin * h))
            (right, bottom) = (int(xmax * w), int(ymax * h))

            label = LABELS.get(classes[i], 'N/A')
            detected_labels.append(label)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({int(scores[i]*100)}%)', (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not detected_labels:
        detected_labels = ["no detection"]

    print(f"[LOG] Resolution: {w}x{h}, Input shape: {input_shape}, Detections: {detected_labels}")
    print(f"[TIME] Inference: {inference_time_ms:.2f}ms, Post: {post_time_ms:.2f}ms, Total: {total_time_ms:.2f}ms\n")

    return frame

# STEP 4 Generate Bounding Box
def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_detected = detect_objects(frame_rgb)

        _, jpeg = cv2.imencode('.jpg', frame_detected)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
