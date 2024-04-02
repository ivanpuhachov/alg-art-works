# run this in blender
import bpy
import bpy_extras
import svgwrite


def write_lines_tosvg(
    lines_list,
    filename="lines3.svg",
    width=1000,
    height=1000,
):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    dwg['width'] = '{}px'.format(width)
    dwg['height'] = '{}px'.format(height)
    for line in lines_list:
        polyline = dwg.polyline(line, stroke='black', fill='none')
        dwg.add(polyline)
    dwg.save()


def render_curve(
        curve,
        camera,
        depsgraph,
        scene,
):
    list_of_lines = []
    current_line = []

    points_to_check = list(enumerate(curve.data.splines[0].bezier_points))

    # when cyclic_u we need to double-check the starting point
    if curve.data.splines[0].use_cyclic_u:
        points_to_check.append(points_to_check[0])

    for (i,bp) in points_to_check:
        # https://docs.blender.org/api/current/bpy.types.Scene.html#bpy.types.Scene.ray_cast
        point = bp.co
        direction = (point - camera.location).normalized()
    #    print(f"\n===={i}")
    #    print(f"Point: {point}")
    #    print("Direction:", direction)
        is_intersecting_solids, location, normal, index, object, matrix = bpy.context.scene.ray_cast(
            depsgraph,
            camera.location,
            direction,
        )
        # ray_cast will not compute intersection with Bezier curve, so is_intersecting_solids is True when ray falls on any solid object in your scene
        # let's check if the interection happens after we see this point on a curve
        # if is_intersecting_solids is False we don't need to compute it
        is_intersecting_after = False
        if is_intersecting_solids:
            distance_curve = (point - camera.location).length
            distance_object = (location - camera.location).length
    #        print(f"distance curve: {distance_curve}")
    #        print(f"distance object: {distance_object}")
            is_intersecting_after = distance_object > distance_curve
        
        point_is_visible = (not is_intersecting_solids) or is_intersecting_after
        
        # now we check if the point is in frame
        point_is_in_frame = False
        if point_is_visible:
            # check that it is in frame
            # https://docs.blender.org/api/current/bpy_extras.object_utils.html#bpy_extras.object_utils.world_to_camera_view
            ndc_coords = bpy_extras.object_utils.world_to_camera_view(
                scene, camera, point
            )
    #        print(ndc_coords)
            if (0<= ndc_coords.x <= 1) and (0<= ndc_coords.y <= 1) and (ndc_coords.z >= 0):
                point_is_in_frame = True
                # calculate pixel coordinates in camera
                render_scale = scene.render.resolution_percentage / 100
                point2d = [ndc_coords.x * scene.render.resolution_x * render_scale, (1 - ndc_coords.y) * scene.render.resolution_y * render_scale]
                current_line.append(point2d)
    #            print("point 2d:", point2d)
        if point_is_in_frame:
            continue
        else:
            if len(current_line) > 1:
                list_of_lines.append(current_line)
            current_line = [] 
    #    break

    if len(current_line) > 1:
        list_of_lines.append(current_line)
        current_line = [] 

    print(f"\n--->{len(list_of_lines)}")
    return list_of_lines


if __name__ == "__main__":
    camera = bpy.data.objects.get("Camera")

    # Get the active scene
    scene = bpy.context.scene

    # Get the dependency graph for the scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    list_of_lines = []
    for obj in bpy.data.objects:
#        curve = bpy.data.objects.get("myCurve.003")
        print(obj.name, " :", obj.hide_render)
        if obj.name.startswith("myCurve") and not obj.hide_render:
            list_of_lines += render_curve(curve=obj, camera=camera, depsgraph=depsgraph, scene=scene)

    write_lines_tosvg(
        list_of_lines, 
        filename="out_bagel.svg",
        width=scene.render.resolution_x,
        height=scene.render.resolution_y,
        )