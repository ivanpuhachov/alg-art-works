import svgwrite
from svgpathtools import svg2paths, wsvg, CubicBezier
import numpy as np


def curve_to_circles(uvcurve, N_points=10):
    length = uvcurve.length()
    # print(length)
    m = 1.005 + np.random.rand() * 0.3
    overlap = np.random.rand() * 0.4
    r = length * (m - 1) / ((m**N_points - 1) * (m+1) * (1-overlap))
    coords_s = [(1 - overlap)*(m+1) * r * (m**i - 1) / (m - 1) for i in range(N_points)]
    coords_t = [uvcurve.ilength(s) for s in coords_s]
    coords_uv = [(uvcurve.point(t).imag, uvcurve.point(t).real) for t in coords_t]
    radii = [r*(m**i) for i in range(N_points)]
    return coords_uv, radii

def write_lines_tosvg(
    coords,
    rads,
    filename="circles.svg",
    width=100,
    height=100,
):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    dwg['width'] = '{}mm'.format(width)
    dwg['height'] = '{}mm'.format(height)
    for i in range(len(rads)):
        circle = dwg.circle(center=coords[i], r=rads[i], stroke='black', fill='white', stroke_width=0.5)
        dwg.add(circle)
    dwg.save()

N_points = np.random.randint(30, 60)  # n points on bezier curve
paths, attributes = svg2paths('grass.svg')
# print(paths)
# uvcurve = paths[10]
# print(uvcurve)
allc = []
allr = []
for i in range(len(paths)):
    if paths[i].length() < 1: 
        continue
    else:
        cs, rs = curve_to_circles(uvcurve=paths[i])
        allc.extend(cs)
        allr.extend(rs)

write_lines_tosvg(coords=allc, rads=allr)
