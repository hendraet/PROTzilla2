import math

import numpy as np


def generate_tics(lower_bound, upper_bound, log: bool):
    """
    Generates a dictionary, mapping equally spaced positions for labels, in the interval between min and max

    :param lower_bound: lower bound of the interval to create labels for
    :param upper_bound: upper bound of the interval to create labels for
    :param log: specifies whether the scale is logarithmic, and the labels should be pow 10
    :return: the dictionary
    """
    temp = math.floor(np.log10(upper_bound - lower_bound) / 2)
    step_size = pow(10, temp)
    first_step = math.ceil(lower_bound / step_size) * step_size
    last_step = math.ceil(upper_bound / step_size) * step_size + 3 * step_size
    tickvals = np.arange(first_step, last_step, step_size)
    if log:
        ticktext = np.vectorize(lambda x: millify(pow(10, x)))(tickvals)
    else:
        ticktext = np.vectorize(lambda x: millify(x))(tickvals)
    return dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )


def millify(n) -> str:
    """
    Writes the number n in shortened style with shorthand symbol for every power of 1000

    :param n: the number to be written in shortened style
    :return: a String containing the shortened number
    """
    millnames = ["", "K", "M", "B", "T", "Q", "Q", "S", "S", "O", "N"]
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])

def style_text(text: str, letter_spacing: float, word_spacing: float) -> str:
    """
    Function to style text with letter spacing and word spacing.

    :param text: The text to be styled.
    :param letter_spacing: The amount of letter spacing.
    :param word_spacing: The amount of word spacing.
    :return: HTML formatted string with specified letter and word spacing.
    """
    return f'<span style="letter-spacing:{letter_spacing}pt;word-spacing:{word_spacing}pt;">{text}</span>'


def enhanced_font_size_spacing(enhanced_reading: bool) -> tuple:
    """
    Function to return the font size and spacing for the plotly graphs
    based on the enhanced reading boolean.

    :param enhanced_reading: Boolean to determine if the font size and
    spacing should be increased.
    :return: Tuple of the font size and spacing.
    """
    add_font_size = 0
    add_letter_spacing = 0
    add_word_spacing = 0
    if enhanced_reading:
        add_font_size = 2
        add_letter_spacing = 2.5
        add_word_spacing = 8.75
    return add_font_size, add_letter_spacing, add_word_spacing

def add_spacing(text: str, letter_spacing: float, word_spacing: float) -> str:
    """
    Adds additional spacing to letters and words for the given text.

    :param text: The text to which spacing should be added.
    :param letter_spacing: The amount of spacing to add between letters.
    :param word_spacing: The amount of spacing to add between words.
    :return: The text with added spacing.
    """
    # Adding letter spacing by inserting additional spaces between each character
    spaced_text = f" {' ' * int(letter_spacing)} ".join(text)
    # Adding word spacing by replacing single spaces with more spaces
    #spaced_text = spaced_text.replace(' ', ' ' * int(word_spacing))
    return spaced_text