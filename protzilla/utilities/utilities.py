import base64
import io
import operator
import os
from itertools import groupby
from random import choices
from string import ascii_letters

import psutil


# recipie from https://docs.python.org/3/library/itertools.html
def unique_justseen(iterable, key=None):
    """List unique elements, preserving order. Remember only the element just seen."""
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBcCAD', str.lower) --> A B c A D
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def random_string():
    return "".join(choices(ascii_letters, k=16))


def get_memory_usage():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return f"{round(memory_mb, 1)} MB"


def clean_uniprot_id(uniprot_id):
    if "-" in uniprot_id:
        uniprot_id = uniprot_id.split("-")[0]
    if "_" in uniprot_id:
        uniprot_id = uniprot_id.split("_")[0]
    return uniprot_id


def fig_to_base64(fig):
    """
    Convert a matplotlib figure to base64. This is used to display the figure in the browser.

    :param fig: matplotlib figure
    :type fig: matplotlib.figure.Figure
    :return: base64 encoded image
    :rtype: bytes
    """
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue())


def isBaseEncoded64(base64_string):
    # soruce https://stackoverflow.com/questions/12315398/check-if-a-string-is-encoded-in-base64-using-python#:~:text=All%20you%20need%20to%20do,then%20it%20is%20base64%20encoded.&text=That's%20it!
    try:
        base64.b64decode(base64_string)
        return True
    except (TypeError, ValueError):
        return False


def remove_underscore_and_capitalize(s):
    words = s.split("_")
    words = [word.capitalize() for word in words]
    return " ".join(words)


def replace_spaces_with_underscores_and_lowercase(s):
    modified_string = s.replace(" ", "_")
    modified_string = modified_string.lower()
    return modified_string
