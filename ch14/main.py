#  tensorflow 2.x
a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
c = tf.constant(3, name="c")
z = 2 * (a - b) + c
tf.print("Result: z = ", z)


def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


@tf.function
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


tf.print("Scalar Inputs:", compute_z(1, 2, 3))
tf.print("Rank 1 Inputs:", compute_z([1], [2], [3]))
tf.print("Rank 2 Inputs:", compute_z([[1]], [[2]], [[3]]))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    )
)
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z


tf.print("Rank 1 Inputs:", compute_z([1], [2], [3]))
tf.print("Rank 1 Inputs:", compute_z([1, 2], [2, 4], [3, 6]))
