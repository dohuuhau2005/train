import tensorflow as tf
print("Số lượng GPU nhận diện được: ", len(tf.config.list_physical_devices('GPU')))