
import glob
import os
import pickle
import queue
import random
import threading

from pytils import check


EXTENSION = ".pickle"
# 2147483648 - 1
MAX_BYTES = (2**31) - 1
# 100 MB = 100 * 1024 KB
TARGET_FILE_SIZE = 100 * 1024 * 1024
STREAM_TARGET_FILE_SIZE = 10 * 1024 * 1024
STREAM_MAX_BATCH = 2000


def dump(data, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    if isinstance(data, queue.Queue):
        thread = threading.Thread(target=_dump_stream, args=[data, dir_path])
        thread.daemon = True
        thread.start()
    else:
        _dump(data, dir_path)


def _dump_stream(data, dir_path):
    check.check_instance(data, queue.Queue)
    batch = []
    batch_size = None
    i = 0

    while True:
        item = data.get()

        if item is not None:
            batch += [item]

            # If we're still building out the sample.
            if batch_size is None:
                # Only try to discover the batch_size every so often.
                if len(batch) % 10 == 0:
                    sample_size = len(pickle.dumps(batch))

                    if sample_size > STREAM_TARGET_FILE_SIZE:
                        average = sample_size / float(len(batch))
                        batch_size = max(1, int(STREAM_TARGET_FILE_SIZE / average))

                if batch_size is None and len(batch) == STREAM_MAX_BATCH:
                    # The batch is plenty large enough - just set it here.
                    batch_size = STREAM_MAX_BATCH
            else:
                # The batch_size has been determined.
                while len(batch) > batch_size:
                    bytes_out = pickle.dumps(batch[:batch_size])
                    _write_bytes(bytes_out, dir_path, i)
                    i += 1
                    batch = batch[batch_size:]
        else:
            # The data stream is complete - flush the remaining data.
            if len(batch) > 0:
                bytes_out = pickle.dumps(batch)
                _write_bytes(bytes_out, dir_path, i)

            break


def _dump(data, dir_path):
    check.check_list(data)

    if len(data) > 0:
        sample_size = max(1, int(0.1 * len(data)))
        sample_indices = set()

        while len(sample_indices) < sample_size:
            sample_indices.add(random.randint(0, len(data) - 1))

        sample = [data[index] for index in sample_indices]
        average = len(pickle.dumps(sample)) / float(sample_size)
        batch_size = max(1, int(TARGET_FILE_SIZE / average))

        for i, offset in enumerate(range(0, len(data), batch_size)):
            bytes_out = pickle.dumps(data[offset:offset + batch_size])
            _write_bytes(bytes_out, dir_path, i)
    else:
        _write_bytes(pickle.dumps([]), dir_path, 0)


def _write_bytes(bytes_out, dir_path, index):
    write_path = os.path.join(dir_path, str(index) + EXTENSION)

    # TODO
    #if os.path.exists(write_path):
    #    raise ValueError("cannot overwrite existing file: %s" % write_path)

    with open(write_path, "wb") as fh:
        while len(bytes_out) > 0:
            fh.write(bytes_out[:MAX_BYTES])
            bytes_out = bytes_out[MAX_BYTES:]


def load(dir_path):
    sub_files = os.listdir(dir_path)

    for sub_file in sorted(sub_files, key=lambda item: int(item[:item.index(EXTENSION)])):
        file_path = os.path.join(dir_path, sub_file)
        size = os.path.getsize(file_path)

        with open(file_path, "rb") as fh:
            bytes_in = bytearray(0)

            for i in range(0, size, MAX_BYTES):
                bytes_in += fh.read(MAX_BYTES)

            for item in pickle.loads(bytes_in):
                yield item

