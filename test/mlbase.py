
import logging
import math
import os
from unittest import TestCase

from ml import base
from pytils.invigilator import create_suite


def tests():
    return create_suite(Tests)


class Tests(TestCase):
    def test_finished_epochs(self):
        tp = base.TrainingParameters()
        losses = tp.losses()
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS + 1, losses), (True, "maximum epochs"))
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS, losses), (False, None))
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS - 1, losses), (False, None))

    def test_finished_absolute(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE
        step = base.TrainingParameters.DEFAULT_ABSOLUTE / float(base.TrainingParameters.DEFAULT_WINDOW)
        # Less than
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (True, "absolute convergence"))
        # Not less than
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append((start * 2) - (step * i))

        self.assertEqual(tp.finished(1, losses), (False, None))
        # Not decreasing
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start)

        self.assertEqual(tp.finished(1, losses), (False, None))

    def test_finished_relative(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE * 1000
        step = start * base.TrainingParameters.DEFAULT_RELATIVE * 0.99
        # Less than
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (True, "relative convergence"), losses)
        # Not less than
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start - (step * i * 2))

        self.assertEqual(tp.finished(1, losses), (False, None))
        # Not decreasing
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start)

        self.assertEqual(tp.finished(1, losses), (False, None))

    def test_finished_degrading(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE
        step = start * base.TrainingParameters.DEFAULT_RELATIVE
        # Degradation
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start + (step if (i + 1) % 2 == 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (True, "degradation"))
        # No degradation
        losses = tp.losses()

        for i in range(base.TrainingParameters.DEFAULT_WINDOW):
            losses.append(start + (step if (i + 1) % 3 == 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (False, None))

