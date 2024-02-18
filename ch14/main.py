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

import tensorflow as tf

a = tf.Variable(initial_value=3.14, name="var_a")
print(a)

b = tf.Variable(initial_value=[1, 2, 3], name="var_b")
print(b)

c = tf.Variable(initial_value=[True, False], dtype=tf.bool)
print(c)

d = tf.Variable(initial_value=["abc"], dtype=tf.string)
print(d)

# 訓練不可変数
w = tf.Variable([1, 2, 3], trainable=False)
print(w.trainable)

print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)
print(w.value())

# 乱数生成
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()
tf.print(init(shape=(3,)))
v = tf.Variable(init(shape=(2, 3)))
tf.print(v)


class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)


m = MyModule()
print("All module variabels:", [v.shape for v in m.variables])

print("Trainable variable:", [v.shape for v in m.trainable_variables])
