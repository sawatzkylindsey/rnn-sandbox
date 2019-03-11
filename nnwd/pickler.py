
import pickle
import os

from pytils import check


byte_separator = "d2e40cd5-9437-450e-ab92-740f70c479c5".encode("utf-8")
max_bytes = (2**31) - 1


def dump(data, file_path):
    check.check_iterable(data)
    dirname = os.path.dirname(file_path)

    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, "wb") as fh:
        for d in data:
            bytes_out = pickle.dumps(d)

            for i in range(0, len(bytes_out), max_bytes):
                fh.write(bytes_out[i:i + max_bytes])

            fh.write(byte_separator)


def load(file_path):
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as fh:
        bytes_in = bytearray(0)

        for i in range(0, size, max_bytes):
            bytes_in += fh.read(max_bytes)

            while byte_separator in bytes_in:
                index = bytes_in.index(byte_separator)
                item = pickle.loads(bytes_in[:index])
                bytes_in = bytes_in[index + len(byte_separator):]
                yield item

