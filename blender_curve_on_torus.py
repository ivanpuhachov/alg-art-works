# run this in blender
import bpy
import numpy as np
from mathutils import Vector

N_points = 1000  # n points on bezier curve
N_wraps = 24  # n of wraps around the torus
myalpha = np.arctan(N_wraps)
curve_t_length = 2 * np.pi * np.sqrt(N_wraps**2 + 1) if N_wraps>1 else 2 * np.pi * np.sqrt((1/N_wraps)**2 + 1)

def to_torus(phi, theta, R=2, r=1):
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def curve_to_torus(t, alpha=np.pi / 6, R=2, r=1):
    phi = t * np.sin(alpha)
    theta = t * np.cos(alpha)
    return to_torus(phi=phi, theta=theta, R=R, r=r)

curveData = bpy.data.curves.new('myTorus', type='CURVE')
curveData.dimensions = '3D'
#curveData.fill_mode = 'BOTH'

coords_t = np.linspace(0, curve_t_length, N_points, endpoint=False)

curveData = bpy.data.curves.new(f'curve', type='CURVE')
curveData.dimensions = '3D'
polyline = curveData.splines.new('BEZIER')
polyline.bezier_points.add(len(coords_t)-1)
polyline.use_cyclic_u = True
for i, t in enumerate(coords_t):
    P = Vector(curve_to_torus(t=t, alpha=myalpha))
    polyline.bezier_points[i].co = P
    polyline.bezier_points[i].handle_left = P 
    polyline.bezier_points[i].handle_right = P

# setup the tangents
for i, t in enumerate(coords_t):
    P1 = Vector(curve_to_torus(t=t, alpha=myalpha))
    # direct differentiation is better, but I'm lazy here and use finite differences
    prev_t, next_t = coords_t[i-1], coords_t[(i+1)%N_points]
    P0 = Vector(curve_to_torus(t=prev_t, alpha=myalpha))
    P2 = Vector(curve_to_torus(t=next_t, alpha=myalpha))
    tan = P2 - P0
#    tan = P2 - P0 + Vector(np.random.rand(3).tolist()).normalized()
    tan.normalize()
#    tan_x, tan_y, tan_z = (x2 - x0) / 10, (y2 - y0) / 10, (z2 - z0) / 10
#    tan_x, tan_y, tan_z = (x2 - x0) / 10, (y2 - y0) / 10, (z2 - z0) / 10
    polyline.bezier_points[i].handle_left = P1 - tan / 10
    polyline.bezier_points[i].handle_right = P1 + tan / 10

curveOB = bpy.data.objects.new('myCurve', curveData)
view_layer = bpy.context.view_layer
curveOB = bpy.data.objects.new('myCurve', curveData)
view_layer.active_layer_collection.collection.objects.link(curveOB)