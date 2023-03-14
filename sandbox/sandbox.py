import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(
        math.factorial(i) * math.factorial(n - i))


def bernstein(t, i, n):
    """Bernstein polynom"""
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(t, points):
    """Calculate coordinate of a point in the bezier curve"""
    n = len(points) - 1
    x = y = 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    # return np.ceil(x), np.ceil(y)
    return int(x), int(y)


def bezier_curve_range(n, points):
    """Range of points in a curve bezier"""
    coord_list = []
    for i in range(n):
        t = i / float(n - 1)
        # yield bezier(t, points)
        coord_list.append(bezier(t, points))
    coord_list = np.array(coord_list)
    x = coord_list[:, 0]  # .tolist()
    y = coord_list[:, 1]  # .tolist()
    return x, y


def main():
    x_coords = [0,   0, 100, 100,   0]
    y_coords = [0, 100, 100, 200, 200]
    points = np.stack([x_coords, y_coords]).T

    x, y = bezier_curve_range(100, points)
    plt.plot(x, y)
    pass


if __name__ == "__main__":
    main()
