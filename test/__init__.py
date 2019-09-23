#!/usr/bin/python
# -*- coding: utf-8 -*-

from pytils.invigilator import create_suite


from test import geometry
from test import mlbase
from test import monotonic_paths


def all():
    return create_suite(unit())


def unit():
    return [
        geometry.tests(),
        mlbase.tests(),
        monotonic_paths.tests(),
    ]

