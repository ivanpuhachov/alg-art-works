import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np
import matplotlib.pyplot as plt
import ndjson
import json
from tqdm import tqdm

experiment_name = "donutonionsheepoctopus"

with open(f'logs/{experiment_name}/config.json', 'r') as json_file:
    config = json.load(json_file)

scaling = 9/255

data = np.load(f"logs/{experiment_name}/final_projections.npz")
print(data.files)

drawing_data = dict()
print("Load NDJSON files")
for n in tqdm(data['mynames']):
    with open(f'/home/ivan/datasets/quickdraw/{n}.ndjson') as f:
        drawing_data[n] = ndjson.load(f)[:data['n_to_plot']]
print(drawing_data.keys())
# print(drawing_data['donut'][0].keys())
# print(drawing_data['donut'][0]['drawing'])
# print(len(drawing_data['donut'][0]['drawing']))

dwg = svgwrite.Drawing(f"logs/{experiment_name}/output_tsne.svg", profile='full', size=('350mm', '280mm'))

inkscape = Inkscape(dwg)
layers = [inkscape.layer(label=f"{i}_{n}") for i,n in enumerate(data['mynames'])]
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
                dwg.polyline(points=points.tolist(), stroke='black', fill='none', stroke_width=.2,)
            )

dwg.save()