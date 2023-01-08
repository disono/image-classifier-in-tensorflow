import os

# disable warning msg for no GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# GPU (limit memory usage)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialization of Tensors
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
x = tf.constant([[1, 2, 3], [4, 5, 6]])

x = tf.ones((3, 3))
x = tf.zeros((2, 3))
x = tf.eye(3)  # I for the identity matrix (eye)
x = tf.random.normal((3, 3), mean=0, stddev=1)
x = tf.random.uniform((1, 3), minval=1, maxval=1)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)
print(x)

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])
z = tf.add(x, y)
print(z)

z = tf.subtract(x, y)
print(z)

z = tf.divide(x, y)
print(z)

z = tf.multiply(x, y)
print(z)

z = tf.tensordot(x, y, axes=1)
z = tf.reduce_sum(x * y, axis=0)
print(z)

z = x ** 5
print(z)

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)
print(z)

z = x @ y
print(z)

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])
print(x[::-1])

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])

print(x[0, :])
print(x[0:2, :])

# Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3, 3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
