import tensorflow as tf
import numpy as np
import cv2
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Path ke model dan label
PATH_TO_SAVED_MODEL = "exported-model/saved_model"
PATH_TO_LABELS = "annotations/label_map.pbtxt"
PATH_TO_IMAGES = "images/test"
OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model dan label
print("Memuat model...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print("Model siap.")

# Proses semua gambar dalam folder test
for image_name in os.listdir(PATH_TO_IMAGES):
    image_path = os.path.join(PATH_TO_IMAGES, image_name)
    image_np = cv2.imread(image_path)

    if image_np is None:
        print(f"Gagal membaca gambar: {image_name}")
        continue

    # Konversi BGR (OpenCV) ke RGB (TensorFlow)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Deteksi objek
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    print(f"\n{image_name} - Deteksi skor tertinggi: {scores[:5]}")  # Debug skor deteksi

    # Visualisasi deteksi
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,  # Tetap BGR agar bisa disimpan dengan benar oleh OpenCV
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=0.5,  # bisa diturunkan jadi 0.2 kalau tidak ada deteksi
        agnostic_mode=False
    )

    # Simpan hasil deteksi
    output_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, image_np)
    print(f"Disimpan ke: {output_path}")
