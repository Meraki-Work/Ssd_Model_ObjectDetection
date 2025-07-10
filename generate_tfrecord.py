import os
import io
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
from object_detection.utils import dataset_util

# Ganti dengan nama label di label_map.pbtxt
LABEL_DICT = {
    'cengkeh_matang': 1,
    'cengkeh_mentah': 2,
    'cengkeh_batang': 3
}

def create_tf_example(example_path, annotations_dir):
    with tf.io.gfile.GFile(example_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    filename = os.path.basename(example_path).encode('utf8')
    image_format = b'jpg'  # ganti 'png' jika menggunakan PNG

    xml_path = os.path.join(annotations_dir, os.path.splitext(os.path.basename(example_path))[0] + '.xml')
    
    if not os.path.exists(xml_path):
        print(f"[SKIP] XML tidak ditemukan untuk {example_path}")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in LABEL_DICT:
            print(f"[SKIP] Label tidak dikenal: {label}")
            continue
        bndbox = obj.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)
        classes_text.append(label.encode('utf8'))
        classes.append(LABEL_DICT[label])

    if not classes:
        print(f"[SKIP] Tidak ada anotasi valid di {example_path}")
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_record(images_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    total = 0
    skipped = 0

    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            xml_path = os.path.splitext(image_path)[0] + ".xml"
            if not os.path.exists(xml_path):
                print(f"[SKIP] {filename} tidak punya file XML")
                skipped += 1
                continue

            tf_example = create_tf_example(image_path, images_dir)
            if tf_example:
                writer.write(tf_example.SerializeToString())
                total += 1
            else:
                skipped += 1

    writer.close()
    print(f"\n✅ TFRecord selesai dibuat di: {output_path}")
    print(f"   Total berhasil   : {total}")
    print(f"   Total dilewati   : {skipped}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', help='Path ke folder gambar dan XML')
    parser.add_argument('--output_path', help='Path untuk menyimpan file .record output')
    args = parser.parse_args()

    create_record(args.images_dir, args.output_path)
