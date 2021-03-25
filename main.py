import tensorflow as tf

with tf.device('/CPU:0'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)