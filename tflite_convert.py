import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("exported-model/saved_model")

# Aktifkan ops TensorFlow agar bisa memuat operasi seperti StridedSlice, map_fn, dll
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Flex ops (wajib untuk model TF2 SSD)
]

# Allow custom ops (tidak semua ops TF bisa di-TFLite-kan natively)
converter.allow_custom_ops = True

# (Opsional) Optimasi ukuran model
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Simpan ke file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model TFLite berhasil dikonversi dengan SELECT_TF_OPS")
