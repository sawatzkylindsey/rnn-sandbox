
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
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS + 1, losses), (True, "maximum epochs"), losses)
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS, losses), (False, None), losses)
        self.assertEqual(tp.finished(base.TrainingParameters.DEFAULT_EPOCHS - 1, losses), (False, None), losses)

    def test_finished_absolute(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE
        step = base.TrainingParameters.DEFAULT_ABSOLUTE / float(tp.window())
        # Less than
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (True, "absolute convergence"), losses)
        # Not less than
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append((start * 2) - (step * i))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)
        # Not decreasing
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start)

        self.assertEqual(tp.finished(1, losses), (False, None), losses)

    def test_non_convergence_absolute(self):
        tp = base.TrainingParameters().convergence(False)
        start = base.TrainingParameters.DEFAULT_ABSOLUTE
        step = base.TrainingParameters.DEFAULT_ABSOLUTE / float(tp.window())
        # Less than - but without convergence
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)

    def test_finished_relative(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE * 1000
        step = start * base.TrainingParameters.DEFAULT_RELATIVE * 0.9
        # Less than
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (True, "relative convergence"), losses)
        # Not less than
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start - (step * i * 2))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)
        # Not decreasing
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start)

        self.assertEqual(tp.finished(1, losses), (False, None), losses)

    def test_non_convergence_relative(self):
        tp = base.TrainingParameters().convergence(False)
        start = base.TrainingParameters.DEFAULT_ABSOLUTE * 1000
        step = start * base.TrainingParameters.DEFAULT_RELATIVE * 0.9
        # Less than - but without convergence
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start - (step * i))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)

    def test_finished_degrading(self):
        tp = base.TrainingParameters()
        start = base.TrainingParameters.DEFAULT_ABSOLUTE
        step = start * base.TrainingParameters.DEFAULT_DEGRADATION * 1.1
        # Degradation
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start + (step if i != 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (True, "degradation"), losses)
        # No degradation
        losses = tp.losses()

        for i in range(tp.window()):
            losses.append(start + (step if (i + 1) % 2 == 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)

    def test_finished_degrading_recovery(self):
        tp = base.TrainingParameters()
        start = tp.absolute()
        step = start * tp.degradation() * 2.1
        step_down = (start * tp.degradation() * 1.1) / (tp.window() * 0.25)
        # Degradation - but with recovery
        losses = tp.losses()
        j = 0

        for i in range(tp.window()):
            if i >= int(tp.window() * 0.75):
                j += 1

            losses.append(start + (step - (step_down * j) if i != 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (False, None), losses)
        # Degradation - but with insufficient recovery
        losses = tp.losses()
        j = 0

        for i in range(tp.window()):
            if i >= int(tp.window() * 0.75) and j == 0:
                j += 1

            losses.append(start + (step - (step_down * j) if i != 0 else 0.0))

        self.assertEqual(tp.finished(1, losses), (True, "degradation"), losses)

