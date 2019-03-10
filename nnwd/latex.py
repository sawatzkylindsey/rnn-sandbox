
import os
import re
import sympy


LATEX_DIR = os.path.join("javascript", "latex")


def generate_png(function):
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)

        if result is not None:
            file_path = os.path.join(LATEX_DIR, "%s.png" % result)

            if not os.path.exists(file_path):
                m = re.match("(tilde_)?(\w+)_(-?\d+)(\^(\d+))?", result)
                base = m.group(2) if m.group(1) is None else r"\tilde{%s}" % m.group(2)

                if m.group(4) is None:
                    sympy.preview(r"$$%s_{%s}$$" % (base, m.group(2)), viewer="file", filename=file_path)
                else:
                    sympy.preview(r"$$%s_{%s}^{%s}$$" % (base, m.group(2), m.group(5)), viewer="file", filename=file_path)

        return result

    return wrapper

