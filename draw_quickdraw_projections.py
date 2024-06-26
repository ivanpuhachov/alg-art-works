import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np
import matplotlib.pyplot as plt
import ndjson
import json
from tqdm import tqdm

palette = ['brown', 'darkgreen', 'darksalmon', 'goldenrod', 'indigo', 'magenta',]

experiment_name = "donutonionappleleaflollipop"

with open(f'logs/{experiment_name}/config.json', 'r') as json_file:
    config = json.load(json_file)

scaling = 9/255

data = np.load(f"logs/{experiment_name}/final_projections.npz")
print(data.files)

drawing_data = dict()
print("Load NDJSON files")
for n in tqdm(data['mynames']):
    with open(f'/home/ivan/datasets/quickdraw/{n}.ndjson') as f:
        drawing_data[n] = ndjson.load(f)
print(drawing_data.keys())
# print(drawing_data['donut'][0].keys())
# print(drawing_data['donut'][0]['drawing'])
# print(len(drawing_data['donut'][0]['drawing']))

dwg = svgwrite.Drawing(f"logs/{experiment_name}/{experiment_name}.svg", profile='full', size=('350mm', '280mm'))

dwg_list = [
    svgwrite.Drawing(f"logs/{experiment_name}/{n}_{experiment_name}.svg", profile='full', size=('350mm', '280mm'))
    for n in data['mynames']
]


inkscape = Inkscape(dwg)
layers = [inkscape.layer(label=f"{i+1}_{n}") for i,n in enumerate(data['mynames'])]
for l in layers:
    dwg.add(l)

for i_name, drawing_name in enumerate(data['mynames']):
    print(drawing_name)
    coords = data[f"{drawing_name}_coords"]

    for idx, coords in zip(data[drawing_name], coords):
        # print(idx)
        # print(coords)
        this_drawing_data = drawing_data[drawing_name][idx]['drawing']
        for coord_pair_list in this_drawing_data:
            points = np.array(coord_pair_list).astype(float).T * scaling + coords
            # print(points.shape)
            layers[
                i_name
            ].add(
                dwg.polyline(points=points.tolist(), stroke=palette[i_name], fill='none', stroke_width=.2,)
            )
            dwg_list[i_name].add(
                dwg.polyline(points=points.tolist(), stroke=palette[i_name], fill='none', stroke_width=.2,)
            )

dwg.save()

for i,n in enumerate(data['mynames']):
    dwg_list[i].save()