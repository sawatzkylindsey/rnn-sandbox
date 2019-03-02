#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import mimetypes
import os
import pdb
import random
from socketserver import ThreadingMixIn
import sys
from threading import Thread
import urllib

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
        return (url.path[1:], data)

    def _set_headers(self, content_type, others={}):
        self.send_response(200)
        self.send_header('Content-type', content_type)

        for key, value in others.items():
            self.send_header(key, value)

        self.end_headers()


def run(port, words, neural_network):
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
    ap.add_argument("--corpus", default="corpora/the-little-prince-v1.txt")
    ap.add_argument("--epochs", default=1000, type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    words, neural_network = domain.create(stream_input(aargs.corpus), aargs.epochs, aargs.verbose)
    run(aargs.port, words, neural_network)


def stream_input(input_file):
    with open(input_file, "r") as fh:
        for line in fh.readlines():
            if line.strip() != "":
                yield line


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

