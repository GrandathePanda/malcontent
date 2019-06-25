import tensorflow as tf
from glob import glob
import pathlib
import textwrap
from datetime import datetime
import re
import numpy as np
from functools import reduce

TYPES = {
  "malware": 0,
  "benign": 1
}
DATA_PATH = pathlib.Path("./processed_data")
VOCAB_STRING = "ARizBSj0CTk1DUl2EVm3FWn4GXo5HYp6IZq7Jar8Kbs9Lct+Mdu/NevOfwPgxQhy"
DICT = list(VOCAB_STRING)
VOCAB_DICT = {v: i for i, v in enumerate(DICT)}

BUFFER_SIZE = 10000
NUM_EPOCHS = 10


def extract(data_path):
    malware = glob(str(data_path / "malware" / "*"))
    benign = glob(str(data_path / "benign" / "*"))

    malware = transform(tf.data.TextLineDataset(filenames=malware), "malware")
    benign = transform(tf.data.TextLineDataset(filenames=benign), "benign")

    dataset = malware.concatenate(benign)
    dataset.shuffle(BUFFER_SIZE)

    return tf.data.Dataset.batch(dataset, 32, drop_remainder=True)


def transform(dataset, _type):
    def labeler(ln):
        return ln, tf.cast(TYPES[_type], tf.int64)

    def tokenizer(ln):
        def _pyfunc(ln):
            def tokenize(memo, itm):
                c = str(chr(itm))
                if c != "=":
                    memo.append(VOCAB_DICT[c])
                return memo

            tokenized = reduce(tokenize, list(ln.numpy()), [])
            return np.asarray(tokenized)

        return tf.py_function(_pyfunc, [ln], Tout=(tf.int64))

    dataset = dataset.map(tokenizer)
    dataset = dataset.map(labeler)

    return dataset


def sample_gen(ds_iter, graph):
    with tf.Session(graph=graph) as sess:
        _next = ds_iter.get_next()
        while True:
            x = sess.run(_next)
            yield x


def _model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=65, input_length=512, output_dim=256))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.CuDNNLSTM(32))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=["acc"])

    return model


def train(model, dataset, graph):
    gen = sample_gen(dataset.make_one_shot_iterator(), graph)
    model.fit(gen, steps_per_epoch=100, epochs=NUM_EPOCHS)
    now = datetime.now()
    model.save(f"malcontent_{datetime.timestamp(now)}.h5")


def main():
    graph = tf.Graph()
    try:
        with graph.as_default():
            dataset = extract(DATA_PATH)
            model = _model()
            train(model, dataset, graph)
    except IndexError:
        pass


if __name__ == "__main__":
    main()
