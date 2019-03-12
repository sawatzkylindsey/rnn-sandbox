
import pickle
import os

from pytils import check


item_separator = "d2e40cd5-9437-450e-ab92-740f70c479c5".encode("utf-8")
# 1073741824 - 1
max_bytes = (2**30) - 1
# 2147483648 - 1
#max_bytes = (2**31) - 1


def dump(data, file_path):
    check.check_iterable(data)
    dirname = os.path.dirname(file_path)

    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, "wb") as fh:
        bytes_out = bytearray(0)

        for d in data:
            bytes_out += pickle.dumps(d)
            bytes_out += item_separator

            if len(bytes_out) >= max_bytes:
                fh.write(bytes_out[:max_bytes])
                bytes_out = bytes_out[max_bytes:]

        fh.write(bytes_out)


def load(file_path):
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as fh:
        bytes_in = bytearray(0)

        for i in range(0, size, max_bytes):
            bytes_in += fh.read(max_bytes)

            while item_separator in bytes_in:
                index = bytes_in.index(item_separator)
                item = pickle.loads(bytes_in[:index])
                bytes_in = bytes_in[index + len(item_separator):]
                yield item

