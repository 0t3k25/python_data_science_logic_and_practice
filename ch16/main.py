import tensorflow as tf

ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)
for ex in ds_text_encoded.take(5):
    print("{} -> {}".format(ex.numpy(), char_array[ex.numpy()]))

# predict next word
seq_length = 40
chunk_size = seq_length + 1

ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)


## define the function for splitting x & y
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


ds_sequences = ds_chunks.map(split_input_target)

## inspection:
for example in ds_sequences.take(2):
    print(" Input (x):", repr("".join(char_array[example[0].numpy()])))
    print("Target (y):", repr("".join(char_array[example[1].numpy()])))
    print()

BATCH_SIZE = 64
BUFFER_SIZE = 10000
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# モデルの構築
def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


# 訓練パラメータ調節
charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512
tf.random.set_seed(1)
model = build_model(
    vocab_size=charset_size, embedding_dim=embedding_dim, rnn_units=rnn_units
)
model.summary()

# モデルの訓練
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(ds, epochs=20)

tf.random.set_seed(1)
logits = [[1.0, 1.0, 1.0]]
# logits = [[1.0,1.0,3.0]]
print("Probabilities:", tf.math.softmax(logits).numpy()[0])
samples = tf.random.categorical(logits=logits, num_samples=10)
tf.print(samples.numpy())

tf.random.set_seed(1)
logits = [[1.0, 1.0, 3.0]]
print("Probabilities:", tf.math.softmax(logits).numpy()[0])
samples = tf.random.categorical(logits=logits, num_samples=10)
tf.print(samples.numpy())


def sample(
    model, starting_str, len_generated_text=500, max_input_length=40, scale_factor=1.0
):
    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.reset_states()
    for i in range(len_generated_text):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical(scaled_logits, num_samples=1)
        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()
        generated_str += str(char_array[new_char_indx])
        new_char_indx = tf.expand_dims([new_char_indx], 0)
        encoded_input = tf.concat([encoded_input, new_char_indx], axis=1)
        encoded_input = encoded_input[:, -max_input_length:]
    return generated_str


# 新しい文章の作成
tf.random.set_seed(2)
print(sample(model, starting_str="The island"))

logits = np.array([1.0, 1.0, 3.0])
print("Probabilities before scaling:", tf.math.softmax(logits).numpy())
print("Probabilities after scaling 2.0:", tf.math.softmax(2.0 * logits).numpy())
print("Probabilities after scaling 0.5:", tf.math.softmax(0.5 * logits).numpy())
print("Probabilities after scaling 0.1:", tf.math.softmax(0.1 * logits).numpy())

tf.random.set_seed(2)
print(sample(model, starting_str="The island", scale_factor=2.0))

tf.random.set_seed(1)
print(sample(model, starting_str="The island", scale_factor=2.0))

tf.random.set_seed(1)
print(sample(model, starting_str="The island", scale_factor=0.5))
