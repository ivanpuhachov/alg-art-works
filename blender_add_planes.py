import bpy
import numpy as np


planerange = (0, 2)
n_planes = 4
for z in np.linspace(planerange[0], planerange[1], num=n_planes, endpoint=True):
    bpy.ops.mesh.primitive_plane_add(size=4, enter_editmode=False, align='WORLD', location=(0, 0, z), scale=(1, 1, 1))
    bpy.context.object.name = f"plane{z:0.2f}"
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Cube"]
    bpy.context.object.modifiers["Boolean"].operation = 'INTERSECT'
    bpy.ops.object.modifier_apply(modifier="Boolean")

