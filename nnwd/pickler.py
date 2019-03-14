
import glob
import pickle
import os
import random

from pytils import check


# 2147483648 - 1
max_bytes = (2**31) - 1
# 100 MB = 100 * 1024 KB
target_file_size = 100 * 1024 * 1024


def dump(data, file_path):
    check.check_list(data)
    dirname = os.path.dirname(file_path)

    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    if len(data) > 0:
        sample_size = max(1, int(0.1 * len(data)))
        sample_indices = set()

        while len(sample_indices) < sample_size:
            sample_indices.add(random.randint(0, len(data) - 1))

        sample = [data[index] for index in sample_indices]
        average = len(pickle.dumps(sample)) / float(sample_size)
        batch_size = max(1, int(target_file_size / average))

        for i, offset in enumerate(range(0, len(data), batch_size)):
            bytes_out = pickle.dumps(data[offset:offset + batch_size])

            with open(file_path + (".%d" % i), "wb") as fh:
                while len(bytes_out) > 0:
                    fh.write(bytes_out[:max_bytes])
                    bytes_out = bytes_out[max_bytes:]
    else:
        with open(file_path + ".0", "wb") as fh:
            fh.write(pickle.dumps([]))


def load(file_path):
    sub_files = glob.glob(file_path + ".*")

    for sub_file in sorted(sub_files, key=lambda item: int(item[item.rindex(".") + 1:])):
        size = os.path.getsize(sub_file)

        with open(sub_file, "rb") as fh:
            bytes_in = bytearray(0)

            for i in range(0, size, max_bytes):
                bytes_in += fh.read(max_bytes)

            for item in pickle.loads(bytes_in):
                yield item

