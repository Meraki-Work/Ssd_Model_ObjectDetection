Install virtual enviroment

TensorFlow: 2.10.0
TensorFlow Addons: 0.21.0
TensorFlow Datasets: 4.9.0
TensorFlow Hub: 0.16.1
TensorFlow I/O: 0.31.0
TF Models Official: 2.10.1
Keras: 2.10.0
Scikit-learn: 1.3.2
SciPy: 1.10.1
NumPy: 1.24.4

Untuk Menjalankan training
python model_main_tf2.py --pipeline_config_path=pretrained_model/pipeline.config --model_dir=training --alsologtostderr

Untuk Export tfrecord
python generate_tfrecord.py --images_dir images/train --output_path annotations/test.record

Untuk Export tflite
python exporter_main_v2.py --pipeline_config_path=pretrained_model/pipeline.config --trained_checkpoint_dir=training --output_directory=exported-model

