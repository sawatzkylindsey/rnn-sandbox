
import pickle
import os


max_bytes = (2**31) - 1


def dump(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    bytes_out = pickle.dumps(data)

    with open(file_path, "wb") as fh:
        for i in range(0, len(bytes_out), max_bytes):
            fh.write(bytes_out[i:i + max_bytes])


def load(file_path):
    bytes_in = bytearray(0)
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as fh:
        for i in range(0, size, max_bytes):
            bytes_in += fh.read(max_bytes)

    return pickle.loads(bytes_in)

