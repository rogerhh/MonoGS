import matplotlib.pyplot as plt

def get_color(idx):
    """
    Get the color for a given index using a consistent color order.
    The colors are defined in the get_color_order function.
    """
    colors = get_color_order()
    return colors[idx % len(colors)]

def get_color_order():
    colors = []
    colors.append((0, 0, 1))  # blue
    colors.append((1, 0.647, 0))  # orange
    colors.append((0, 0.5, 0))  # green
    colors.append((1, 0, 0))  # red
    colors.append((0.5, 0, 0.5))  # purple
    colors.append((0.545, 0.271, 0.075))  # brown
    colors.append((1, 0.75, 0.8))  # pink
    colors.append((0.5, 0.5, 0.5))  # gray
    colors.append((0.5, 0.5, 0))  # olive
    colors.append((0, 1, 1))  # cyan
    colors.append((1, 0, 1))  # magenta
    colors.append((0.5, 1, 0))  # lime
    colors.append((0, 0.5, 0.5))  # teal
    colors.append((0, 0, 0.5))  # navy
    colors.append((0.5, 0, 0))  # maroon
    colors.append((1, 0.843, 0))  # gold
    colors.append((1, 0.5, 0))  # coral
    colors.append((0.98, 0.5, 0.447))  # salmon
    colors.append((0.941, 0.902, 0.549))  # khaki
    colors.append((0.867, 0.627, 0.867))  # plum
    colors.append((0.294, 0, 0.51))  # indigo

    return colors

