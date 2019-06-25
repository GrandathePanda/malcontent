from glob import glob
import pathlib
from base64 import b64encode
import textwrap

DATA_PATH = pathlib.Path("./data_raw")
EXPORT_PATH = pathlib.Path("./processed_data")


def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def split_file(file):
    return filter(lambda ln: len(ln) == 514, map(lambda ln: ln + "\n", chunkstring(file, 512)))


def transform(files):
    for file in files:
        with open(file, 'rb') as f:
            contents = b64encode(f.read()).decode("utf-8")
            yield split_file(contents)


def export(generator, _type):
    try:
        i = 0
        while True:
            text = next(generator)
            with open(EXPORT_PATH/_type/f"file_{i}", "w") as f:
                f.writelines(text)
            i += 1
    except StopIteration:
        pass


def main():
    malware = glob(str(DATA_PATH / "malware" / "*"))
    benign = glob(str(DATA_PATH / "benign" / "*"))

    m_transformer = transform(malware)
    b_transformer = transform(benign)

    export(m_transformer, "malware")
    export(b_transformer, "benign")


if __name__ == "__main__":
    main()
