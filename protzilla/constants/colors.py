import re

PROTZILLA_DISCRETE_COLOR_SEQUENCE = [
    "#4A536A",  # blue
    "#87A8B9",  # light blue
    "#CE5A5A",  # red
    "#8E3325",  # dark red
    "#E2A46D"   # orange
]

PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE = [
    "#4A536A",  # blue
    "#CE5A5A",  # red
    "#87A8B9",  # light blue
    "#a1a5b2",  # grey
    "#E2A46D"   # orange
]

PROTAN_DISCRETE_COLOR_SEQUENCE = [
    "#4A536A",  # blue
    "#daa217",  # yellow
    "#a8b3ef",  # light blue
    "#a1a5b2",  # grey

]
# justify how colors come about
DEUTAN_DISCRETE_COLOR_SEQUENCE = [
    "#4A536A",  # blue
    "#FF911E",  # yellow
    "#A4B6EF",  # light blue
    "#a1a5b2"   # grey

]

TRITAN_DISCRETE_COLOR_SEQUENCE = [
    "#48565d",  # blue
    "#f48e9b",  # pink
    "#99bbca",  # light blue
    "#a1a5b2"   # grey
]

MONOCHROMATIC_DISCRETE_COLOR_SEQUENCE = [
    "#929292",  # medium grey
    "#4C4C4C",  # dark grey
    "#949494",  # light grey
    "#C5C5C5",  # grey
    "#333333"   # black
]


def get_color_sequence(colors: str):
    global PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
    color_sequences = {
        "standard": PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE,
        "protan": PROTAN_DISCRETE_COLOR_SEQUENCE,
        "deutan": DEUTAN_DISCRETE_COLOR_SEQUENCE,
        "tritan": TRITAN_DISCRETE_COLOR_SEQUENCE,
        "monochromatic": MONOCHROMATIC_DISCRETE_COLOR_SEQUENCE
    }

    if colors in color_sequences:
        PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE = color_sequences[colors]


def set_custom_sequence(custom_color_sequence: str):
    global PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE

    if not (is_valid_hex_color_pair(custom_color_sequence)):
        raise ValueError("Invalid hex color pair")

    custom_color_list = custom_color_sequence.split(",")
    PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE = custom_color_list


def is_valid_hex_color_pair(s):
    hex_color_pattern = r'#[0-9a-fA-F]{6}'
    pattern = re.compile(f'^{hex_color_pattern}, {hex_color_pattern}$')
    return bool(pattern.match(s))
