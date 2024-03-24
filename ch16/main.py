# predict next word
seq_length = 40
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)


# xとyを分離するための関数を定義
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


ds_sequences = ds_chunks.map(split_input_target)

for example in ds_sequences.take(2):
    print("Input(x):", repr("".join(char_array[example[0].numpy()])))
    print("Target(y):", repr("".join(char_array[example[0].numpy()])))
