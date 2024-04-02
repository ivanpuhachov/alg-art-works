This repo started as an attempt to do level plots in Python

*** 
# Perlin noise circles field


***
# No banana
Inspired by marching cubes algo - we work with negative space.
![no banana example](demo/nobanana.png)
see `nobanana.py` 
***
# QuickDraw 2d projection
![latent space](demo/latent.png)
Dependencies:
 * https://github.com/stefankoegl/kdtree
```
pip install kdtree
```
* `quickdraw_ae.py` and `mnist_vae_test.py` for AE and VAE attempts
* `vis_latents.py` to project latent space on 2d (tsne and umap)
* `sample_projections.py` to downsample dense 2d projections