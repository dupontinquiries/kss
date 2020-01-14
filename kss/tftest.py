import tensorflow as tf
import numpy as np

from datetime import datetime

#a = np.array([[1, 2, 3], [4, 5, 6]])
#b = np.array([1, 2, 3])

tf.test.is_gpu_available()

a = np.random.rand(100000, 100000)
b = np.random.rand(100000, 100000)

print("Init complete");

start = datetime.now()

result = tf.math.multiply(a , b)

end = datetime.now()

print('result = {0},\n time elapsed = {1}' \
.format(result, end-start))
