
import markdown
from latex import MarkdownLatex


EXTENSIONS = [
    "extra",
    "smarty",
    MarkdownLatex(),
]


def render(latex_snippet):
    markdown.markdown(latex_snippet, extensions=EXTENSIONS, output_format="html5")

