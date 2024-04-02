from perlin_numpy2d import generate_perlin_noise_2d, generate_fractal_noise_2d
import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svg2paths2, wsvg, Path, CubicBezier, Line, QuadraticBezier, Arc
import svgwrite


def write_paths_to_svg(
        paths_to_save: list,
        save_to: str,
        colors=None,
):
    if colors is None:
        colors = [None for _ in paths_to_save]
    # sorted_paths = sorted(paths_to_save, key=lambda x: x.length(), reverse=True)
    attributes_to_save = [
        {
            'fill': 'none',
            'stroke-width': 0.01,
            'stroke': "rgb({},{},{})".format(*get_tab20_rgb(i) if colors[i] is None else colors[i]),
        }
        for i in range(len(paths_to_save))
    ]
    wsvg(paths_to_save, attributes=attributes_to_save, filename=save_to)


def get_tab20_rgb(ind: int):
    ii = ind % 20
    r, g, b, a = plt.cm.tab10(ii)
    return int(255*r), int(255*g), int(255*b)

if __name__ == "__main__":
    noise_size = 64
    xx = generate_perlin_noise_2d(
        shape=(noise_size, noise_size), 
        res=(1, 1),
    )
    xgrad = 10 * np.stack(np.gradient(f=xx), axis=2)
    print(xgrad.shape)
    yy = generate_perlin_noise_2d(
        shape=(noise_size, noise_size), 
        res=(1, 1),
    )
    ygrad = 10 * np.stack(np.gradient(f=yy), axis=2)

    nn = np.hypot(xx, yy)

    # xx = xx / nn
    # yy = yy / nn

    lines = []
    colors = []

    n_draw_grid = 8


    for (i, ix) in enumerate(np.arange(0, noise_size, noise_size//n_draw_grid)):
        for (j, jy) in enumerate(np.arange(0, noise_size, noise_size//n_draw_grid)):
            startpoint = complex(5*i, 5*j)
            endpoint = startpoint + 10 * complex(xx[ix, jy], yy[ix, jy])
            controlpoint = startpoint + 10 * complex(xgrad[ix, jy, 0], xgrad[ix, jy, 1]) - 10 * complex(ygrad[ix, jy, 0], ygrad[ix, jy, 1])
            lines.append(
                Path(
                    QuadraticBezier(
                        start=startpoint,
                        control=controlpoint,
                        end=endpoint,
                        )
                    )
            )
            colors.append(
                (128 + 60 * xx[ix, jy], 128, 128 + 60 * yy[ix, jy])
            )

    write_paths_to_svg(paths_to_save=lines, save_to="grass.svg")

    # plt.figure(figsize=(12,5))
    # plt.subplot(121)
    # plt.imshow(xx)
    # plt.colorbar()
    # plt.subplot(122)
    # # plt.imshow(yy)
    # plt.imshow(xgrad[...,0])
    # plt.colorbar()
    # plt.show()