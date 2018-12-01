
from pytils.invigilator import create_suite

from test import geometry


def all():
    return create_suite(unit())


def unit():
    return [
        geometry.tests(),
    ]

