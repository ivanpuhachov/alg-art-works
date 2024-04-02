from shapely.geometry import Point, Polygon
from svgwrite import Drawing, shapes, rgb
from svgpathtools import svg2paths
import numpy as np
from shapely import affinity
from svgwrite.extensions import Inkscape


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return rgb(*(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))


def create_rounded_rectangle(center_x, center_y, width, height, corner_radius):
    half_width = width / 2
    half_height = height / 2
    radius = min(half_width, half_height, corner_radius)
    
    left = center_x - half_width
    right = center_x + half_width
    top = center_y + half_height
    bottom = center_y - half_height
    
    top_left = Point(left + radius, top - radius).buffer(radius)
    top_right = Point(right - radius, top - radius).buffer(radius)
    bottom_left = Point(left + radius, bottom + radius).buffer(radius)
    bottom_right = Point(right - radius, bottom + radius).buffer(radius)
    
    vertices = top_left.exterior.coords[32:49]
    vertices.extend(top_right.exterior.coords[48:])
    vertices.extend(bottom_right.exterior.coords[:17])
    vertices.extend(bottom_left.exterior.coords[16:33])

    rectangle = Polygon(
        vertices
    )
    
    return rectangle

def create_random_triangle(mmin=1, mmax=3):
    xxs = np.random.rand(3,2) * (mmax - mmin) + mmin
    return Polygon(
        xxs
    )

def create_random_banana(points=100):
    paths, attributes = svg2paths('banana.svg')
    curve = paths[0]
    xxs = [[curve.point(i/points).real, curve.point(i/points).imag] for i in range(points)]
    poly =  Polygon(
        xxs
    )
    return affinity.rotate(poly, angle=np.random.randint(60))


if __name__ == "__main__":
    purple = hex_to_rgb("#4f186b")
    blue = hex_to_rgb("#3e4db4")
    ruby = hex_to_rgb("#91144e")
    red = hex_to_rgb("#ea1f25")
    yellow = hex_to_rgb("#f1ca00")
    brown = hex_to_rgb("#ad6d37")

    # palette = [purple, blue, ruby, red, yellow, brown]
    palette = [blue, ruby, yellow]

    banana = create_random_banana()
    bbox_banana = banana.envelope
    x, y = bbox_banana.exterior.coords.xy
    banana = affinity.translate(banana, xoff=-x[0], yoff=-y[0])
    bbox_banana = banana.envelope
    x, y = bbox_banana.exterior.coords.xy
    print(bbox_banana)

    banana_width = x[1]
    banana_height = y[2]

    L = max(x[1], y[2]) + 10
    prob_skip = np.random.rand() * 0.4
    N = np.random.randint(6,20)
    maxd = 0.5 * L / (2*N + 1)
    mind = maxd / 10
    d = np.random.uniform(low=mind, high=maxd)
    xsize = (L - d) / N - d

    banana = affinity.translate(banana, xoff=d + xsize/2, yoff=d+xsize/2)

    dwg = Drawing(f'result{N}.svg', profile='tiny')
    inkscape = Inkscape(dwg)

    layers = [inkscape.layer(label=f"{i}l") for i in range(len(palette))]
    for l in layers:
        dwg.add(l)

    # circle = Point(0, 0).buffer(3)
    # random_tri = create_random_triangle(mmin=d + xsize/2, mmax=L - d - xsize/2)


    for j in range(int((banana_width) // (d + xsize)) + 2):
        for i in range(int((banana_height) // (d + xsize)) + 2):
            if np.random.rand() < prob_skip:
                continue
            xc, yc = xsize/2 + d + j * (xsize + d), xsize/2 + d + i * (xsize + d)
            rectangle = create_rounded_rectangle(xc, yc, xsize, xsize, xsize/8)
            # dwg.add(shapes.Polygon(list(rectangle.exterior.coords)))

            result = rectangle.difference(banana)
            lchoice = np.random.randint(len(palette))
            color = palette[lchoice]
            if result.geom_type == 'Polygon':
                polygons = [result]  
                for polygon in polygons:
                    if len(polygon.exterior.coords) > 0:
                        # dwg.add(
                        layers[lchoice].add(
                            shapes.Polygon(list(polygon.exterior.coords), fill=color, stroke=color, stroke_width=0.2)
                            )
            else:
                continue
    # dwg.add(shapes.Polygon(list(banana.exterior.coords), fill="none", stroke="black", stroke_width=0.2))
    # Save the SVG file
    dwg.save()

    print(f"\n\nresult{N}.svg")