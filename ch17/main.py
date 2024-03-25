# gpuの確認
import tensorflow as tf

print(tf.__version__)

print("GPU Availabel:", tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    device_name = tf.test.gpu_device_name()
else:
    device_name = "/CPU:0"

print(device_name)
