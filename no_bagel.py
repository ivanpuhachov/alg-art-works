from nobanana import create_rounded_rectangle, hex_to_rgb
import numpy as np
import svgwrite
from svgwrite.extensions import Inkscape


if __name__ == "__main__":
    purple = hex_to_rgb("#4f186b")
    blue = hex_to_rgb("#3e4db4")
    ruby = hex_to_rgb("#91144e")
    red = hex_to_rgb("#ea1f25")
    yellow = hex_to_rgb("#f1ca00")
    brown = hex_to_rgb("#ad6d37")

    # palette = [purple, blue, ruby, red, yellow, brown]
    palette = [blue, ruby, yellow]

    # Create an SVG drawing
    dwg = svgwrite.Drawing('output_text.svg', profile='full')
    inkscape = Inkscape(dwg)
    layers = [inkscape.layer(label=f"{i}l") for i in range(len(palette))]
    layers.append(inkscape.layer("text"))
    for l in layers:
        dwg.add(l)
    

    # Add text with random rotation
    w, h = 220, 60

    offset = 20
    text_anchor = (offset, h + offset)
    text = dwg.text(
        'Bagel', 
        insert=text_anchor,
        fill='white',
        stroke='white',
        )
    rot_angle = np.random.randint(-20, 45)
    # rot_angle = 0
    rot_angle_rad = rot_angle * 2 * np.pi / 360
    text.rotate(rot_angle, center=text_anchor)
    text.update({'font-family': 'Rockwell', 'font-size': '60pt'})
    layers[-1].add(text)

    bb_w = h * np.sin(rot_angle_rad) + w* np.cos(rot_angle_rad) + 2*offset
    bb_h = h * np.cos(rot_angle_rad) + w* np.sin(rot_angle_rad) + 2*offset

    prob_skip = np.random.rand() * 0.2

    L = max(bb_w, bb_h)
    N = np.random.randint(6,10)
    # d is a gap between squares
    maxd = 0.5 * L / (2*N + 1) 
    mind = maxd / 10
    d = np.random.uniform(low=mind, high=maxd)
    xsize = (L - d) / N - d

    i, j=0, 0
    while (j< bb_w / (d + xsize)) and (j < N):
        i = 0
        while (i< bb_h / (d + xsize)) and (i < N):
            if np.random.rand() < prob_skip:
                i+=1
                continue
            xc, yc = d + j * (xsize + d), d + i * (xsize + d)
            lchoice = np.random.randint(len(palette))
            color = palette[lchoice]
            rectangle = svgwrite.shapes.Rect(
                insert=(
                    xc,yc
                ), 
                size=(
                    xsize , # width
                    xsize # height
                ), 
                rx=xsize/8, 
                ry=xsize/8, 
                fill=color, stroke=color, stroke_width=.2,
                )
            layers[lchoice].add(
                rectangle
            )
            i += 1
        j += 1
            
    bounding_box_rect =  svgwrite.shapes.Rect(
        insert=(
            0,0
        ), 
        size=(
            bb_w , # width
            bb_h # height
        ), 
        rx=None, 
        ry=None, 
        fill='none', stroke='black', stroke_width=2,
        )

    layers[-1].add(bounding_box_rect)
    # Save the SVG file
    dwg.save()