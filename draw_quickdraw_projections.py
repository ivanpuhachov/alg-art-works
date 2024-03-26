import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np
import matplotlib.pyplot as plt
import ndjson

scaling = 6/255

data = np.load("final_projections.npz")
print(data.files)

drawing_data = dict()
print("Load NDJSON files")
for n in data['mynames']:
    with open(f'/home/ivan/datasets/quickdraw/{n}.ndjson') as f:
        drawing_data[n] = ndjson.load(f)[:data['n_to_plot']]
print(drawing_data.keys())
print(drawing_data['donut'][0].keys())
print(drawing_data['donut'][0]['drawing'])
print(len(drawing_data['donut'][0]['drawing']))


plt.figure()
for name in data['mynames']:
    coords = data[f"{name}_coords"]
    # print(coords.shape)
    plt.scatter(coords[:, 0], coords[:, 1], label=name)
plt.legend()
plt.savefig('test.png')
plt.close()

dwg = svgwrite.Drawing('output_tsne.svg', profile='full')
inkscape = Inkscape(dwg)
layers = [inkscape.layer(label=f"{i}_{n}") for i,n in enumerate(data['mynames'])]
for l in layers:
    dwg.add(l)

for i_name, drawing_name in enumerate(data['mynames']):
    print(drawing_name)
    coords = data[f"{drawing_name}_coords"]

    for idx, coords in zip(data[drawing_name], coords):
        # print(idx)
        print(coords)
        this_drawing_data = drawing_data[drawing_name][idx]['drawing']
        for coord_pair_list in this_drawing_data:
            points = np.array(coord_pair_list).astype(float).T * scaling + coords
            print(points.shape)
            layers[
                i_name
            ].add(
                dwg.polyline(points=points.tolist(), stroke='black', fill='none', stroke_width=.2,)
            )

dwg.save()