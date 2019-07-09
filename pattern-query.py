
from argparse import ArgumentParser
import json
import logging
import os
import sys

from nnwd import domain
from nnwd import sequential
from nnwd.models import Predicates

from pytils.log import setup_logging, teardown, user_log


@teardown
def main():
    ap = ArgumentParser(prog="pattern-query")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("--query-dir", default=None)
    ap.add_argument("--db-kind", choices=["postgres", "sqlite"])
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    aargs = ap.parse_args(sys.argv[1:])
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    query_engine = domain.QueryEngine(lstm, "moot", "postgres")

    # Hard coded - fiddel with manually
    predicates = Predicates([
        {"cells-0": {32: (0.5, None)}},
        {"cells-0": {32: (0.5, "lt")}}
    ])
    result = query_engine.find(0.1, predicates)
    dump = json.dumps(result.as_json())
    logging.debug("%s\n  %s" % (predicates.as_strs(), dump))
    print(dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())

