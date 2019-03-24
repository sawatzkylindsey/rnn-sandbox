#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import bz2
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import mimetypes
import nltk
import os
import pdb
import random
import re
from socketserver import ThreadingMixIn
import sys
# Not used by this module, but loading this up-front seems to be avoiding some very odd threading dealock between the server process and the background setup processes.
import tensorflow
import time
from threading import Thread
import urllib

from ml import nlp
from nnwd import domain
from nnwd import errorhandler
from nnwd import errors
from nnwd import handlers
from pytils.log import setup_logging, user_log


class ServerHandler(BaseHTTPRequestHandler):
    @errorhandler.safely
    def do_GET(self):
        (path, data) = self._read_request()
        logging.debug("GET %s: %s" % (path, data))
        handler = path.replace("/", "_")

        if handler in self.server.handlers:
            out = self.server.handlers[handler].get(data)

            if out == None:
                out = {}

            self._set_headers("application/json")

            if hasattr(out, "as_json"):
                out = out.as_json()

            self._write_content(json.dumps(out))
        else:
            file_path = os.path.join(".", "javascript", path)

            # Some systems (like eccc-nll.bigdata.sfu.ca) allow for relative paths to pass through urllib.
            # I checked the versions of that library, and they are the same even though this problem doesn't exist on my osx.
            # Alas, make sure the constructed path is a subpath of the server's javascript directory.
            if not os.path.abspath(file_path).startswith(os.path.abspath(os.path.join(".", "javascript"))):
                raise errors.NotFound(path)

            if os.path.exists(file_path) and os.path.isfile(file_path):
                mimetype, _ = mimetypes.guess_type(path)

                if mimetype is None:
                    mimetype = "text/plain"

                self._set_headers(mimetype)
                encode = "text" in mimetype
                self._write_file(file_path, encode)
            else:
                raise errors.NotFound(path)

    def _write_file(self, file_path, encode=True):
        if encode:
            with open(file_path, "r") as fh:
                self._write_content(fh.read())
        else:
            with open(file_path, "rb") as fh:
                self.wfile.write(fh.read())

    def _write_content(self, content):
        self.wfile.write(content.encode("utf-8"))

    def _read_request(self):
        url = urllib.parse.urlparse(self.path)
        data = urllib.parse.parse_qs(url.query)
        return (urllib.parse.unquote(url.path[1:]), data)

    def _set_headers(self, content_type, others={}):
        self.send_response(200)
        self.send_header('Content-type', content_type)

        for key, value in others.items():
            self.send_header(key, value)

        self.end_headers()


def run(port, words, neural_network, query_engine):
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        pass

    #patch_Thread_for_profiling()
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, ServerHandler)
    httpd.daemon_threads = True
    httpd.handlers = {
        "echo": handlers.Echo(),
        "weight-explain": handlers.WeightExplain(neural_network),
        "weights": handlers.Weights(neural_network),
        "weight-detail": handlers.WeightDetail(neural_network),
        "words": handlers.Words(words.labels()),
        "sequence-matches": handlers.SequenceMatches(query_engine),
        "sequence-matches-estimate": handlers.SequenceMatchesEstimate(query_engine),
    }
    user_log.info('Starting httpd %d...' % port)
    httpd.serve_forever()


def main(argv):
    ap = ArgumentParser(prog="server")
    ap.add_argument("--verbose", "-v",
                    default=False,
                    action="store_true",
                    help="Turn on verbose logging.")
    ap.add_argument("-p", "--port", default=8888, type=int)
    ap.add_argument("--epochs", default=2, type=int)
    ap.add_argument("-s", "--save-dir", default=".resume")
    ap.add_argument("--tag", default=False, action="store_true")
    ap.add_argument("task", help="Either 'sa' or 'lm'.")
    ap.add_argument("form", help="The appropriate selection of: ['stanford', 'raw', 'pos'].")
    ap.add_argument("corpus_paths", nargs="+")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    if aargs.task == "sa":
        assert aargs.form == "stanford"
        words, neural_network = domain.create_sa((aargs.task, aargs.form), lambda: stream_input_stanford(aargs.corpus_paths[0]), aargs.epochs, aargs.verbose, aargs.save_dir)
    elif aargs.task == "lm":
        assert aargs.form == "raw" or aargs.form == "pos"
        result_form = aargs.form if not aargs.tag else "pos"
        words, neural_network = domain.create_lm((aargs.task, result_form), lambda: stream_input_text(aargs.corpus_paths, aargs.form), aargs.epochs, aargs.verbose, aargs.save_dir)
    else:
        raise ValueError("Unknown task: %s" % aargs.task)

    # Also precautionary - the the neural_network start setting up before kicking off the server.
    # The server can't do anything anyways until the neural_network is ready to handle requests.
    time.sleep(5)
    query_engine = domain.QueryEngine()
    run(aargs.port, words, neural_network, query_engine)


POS_MAP = {
    "CC": "CC",
    "CD": "CD",
    "DT": "DT",
    "EX": "EX",
    "FW": "FW",
    "IN": "IN",
    "JJ": "JJ",
    "JJR": "JJR",
    "JJS": "JJS",
    "LS": "LS",
    "MD": "MD",
    "NN": "NN",
    "NNS": "NNS",
    "NNP": "NNP",
    "NNPS": "NNPS",
    "PDT": "PDT",
    "POS": "POS",
    "PRP": "PRP",
    "PRP$": "PRP$",
    "RB": "RB",
    "RBR": "RBR",
    "RBS": "RBS",
    "RP": "RP",
    "SYM": "SYM",
    "TO": "TO",
    "UH": "UH",
    "VB": "VB",
    "VBD": "VBD",
    "VBG": "VBG",
    "VBN": "VBN",
    "VBP": "VBP",
    "VBZ": "VBZ",
    "WDT": "WDT",
    "WP": "WP",
    "WP$": "WP$",
    "WRB": "WRB",
    ".": "PUNCT",
    ",": "PUNCT",
    "``": "PUNCT",
    "''": "PUNCT",
    ":": "PUNCT",
    ";": "PUNCT",
    "(": "PUNCT",
    ")": "PUNCT",
    "$": "PUNCT",
    "!": "PUNCT",
    "?": "PUNCT",
}
BAD_TAGS = {}


def stream_input_text(input_files, form):
    for input_file in input_files:
        opener = lambda: open(input_file, "r") if not input_file.endswith("bz2") else bz2.BZ2File(input_file)

        with opener() as fh:
            for line in fh.readlines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if line.strip() != "":
                    if form == "raw":
                        for sentence in nlp.split_sentences(line):
                            tagged = nltk.pos_tag(sentence)
                            sequence = []

                            for item in tagged:
                                word, tag = item

                                if tag in POS_MAP:
                                    pos = POS_MAP[tag]

                                    if pos == "SYM" or pos == "PUNCT":
                                        if word == "(" or word == ")":
                                            sequence += [(word, pos)]
                                        else:
                                            pass
                                    else:
                                        sequence += [(word, pos)]
                                elif tag not in BAD_TAGS:
                                    BAD_TAGS[tag] = None
                                    print(tag)

                            yield sequence + [(".", "PUNCT")]
                    else:
                        sequence = []

                        for item in re.split("[()]", line):
                            pair = item.strip().split(" ")

                            if len(pair) == 2:
                                tag = pair[0]

                                if tag in POS_MAP:
                                    word = pair[1].lower()
                                    pos = POS_MAP[tag]

                                    if pos == "SYM" or pos == "PUNCT":
                                        if word == "(" or word == ")":
                                            sequence += [(word, pos)]
                                        else:
                                            pass
                                    else:
                                        sequence += [(word, pos)]
                                elif tag not in BAD_TAGS:
                                    BAD_TAGS[tag] = None
                                    print(tag)

                        yield sequence + [(".", "PUNCT")]


def stream_input_stanford(stanford_folder):
    sentiments = {}

    with open(os.path.join(stanford_folder, "sentiment_labels.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                phrase_id, sentiment = line.split("|")
                sentiment = sentiment.strip()
                sentiments[phrase_id] = float(sentiment)

    dictionary = {}

    with open(os.path.join(stanford_folder, "dictionary.txt"), "r") as fh:
        for line in fh.readlines():
            line = line.strip()
            index = line.rindex("|")
            phrase = line[:index].lower()
            dictionary[phrase] = sentiments[line[index + 1:]]
            dictionary[phrase + " ."] = sentiments[line[index + 1:]]

    dataset_splits = {}

    with open(os.path.join(stanford_folder, "datasetSplit.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                sentence_id, label = line.split(",")
                label = label.strip()
                dataset_splits[sentence_id] = "train" if label == "1" else ("test" if label == "2" else "dev")

    #train_sentences = []

    with open(os.path.join(stanford_folder, "datasetSentences.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                sentence_id, sentence = line.split("\t")
                sentence = sentence.strip().lower()
                sequence = []

                for word in sentence.split(" "):
                    if word != "." \
                        and word != "," \
                        and word != "``" \
                        and word != "''" \
                        and word != ":" \
                        and word != ";" \
                        and word != "$" \
                        and word != "!" \
                        and word != "?":
                        sequence += [word]

                #if dataset_splits[sentence_id] == "train":
                #    train_sentences += [sequence]

                yield (dataset_splits[sentence_id], sequence + ["."], dictionary[sentence])

    #data_tenth = max(1, int(len(dictionary) / 10.0))

    #for i, phrase_sentiment in enumerate(dictionary.items()):
    #    if i % data_tenth == 0 or i + 1 == len(dictionary):
    #        print("%d%% through" % int((i + 1) * 100 / len(dictionary)))

    #    # Sample at 30% rate.
    #    if random.randint(0, 9) < 3:
    #        phrase, sentiment = phrase_sentiment

    #        if any([phrase in sentence for sentence in train_sentences]):
    #            yield ("train", phrase.split(" "), sentiment)


def patch_Thread_for_profiling():
    Thread.stats = None
    thread_run = Thread.run

    def profile_run(self):
        import cProfile
        import pstats
        self._prof = cProfile.Profile()
        self._prof.enable()
        thread_run(self)
        self._prof.disable()
        (_, number) = self.name.split("-")
        self._prof.dump_stats("Thread-%.3d-%s.profile" % (int(number), "".join([chr(97 + random.randrange(26)) for i in range(0, 2)])))

    Thread.run = profile_run


if __name__ == "__main__":
    main(sys.argv[1:])

