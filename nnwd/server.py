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
from nnwd import data
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


def run_server(port, words, neural_network, query_engine):
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        pass

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
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("buckets_dir")
    ap.add_argument("encoding_dir")
    aargs = ap.parse_args(argv)
    #patch_thread_for_profiling()
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    words = data.get_words(aargs.data_dir)
    neural_network = domain.NeuralNetwork(aargs.data_dir, aargs.sequential_dir, aargs.buckets_dir, aargs.encoding_dir)
    query_engine = domain.QueryEngine()
    run_server(aargs.port, words, neural_network, query_engine)

    try:
        neural_network._background_setup.join()
    except KeyboardInterrupt as e:
        if patched:
            neural_network._background_setup.complete_profile()

        raise e

    return 0


patched = False


def patch_thread_for_profiling():
    global patched
    patched = True
    import cProfile
    import pstats
    Thread.stats = None
    thread_run = Thread.run

    def complete_profile(self):
        self._prof.disable()
        (_, number) = self.name.split("-")
        self._prof.dump_stats("Thread-%.3d-%s.profile" % (int(number), "".join([chr(97 + random.randrange(26)) for i in range(0, 2)])))

    def profile_run(self):
        self._prof = cProfile.Profile()
        self._prof.enable()

        try:
            thread_run(self)
        except Exception as e:
            raise e
        finally:
            self.complete_profile()

    Thread.run = profile_run
    Thread.complete_profile = complete_profile


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

