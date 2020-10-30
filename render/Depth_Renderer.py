#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/render/render_depth.py
# Usage: blender -b -P Depth_Renderer.py [ShapeNet directory] [model list] [output directory] [num scans per model]

import os, sys, bpy, time, mathutils, numpy as np


def random_pose():
    """generate a random camera pose"""
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)  # select the third column, reshape into (3, 1)-vector

    # pose -> 4 * 4
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose


def setup_blender(width, height, focal_length):
    """using blender to rendering a scene"""
    # camera, a class in the bpy
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    # tree.links.new(rl.outputs['Depth'], output.inputs[0])
    tree.links.new(rl.outputs['Z'], output.inputs[0])
    # ref: https://github.com/panmari/stanford-shapenet-renderer/issues/8

    # remove default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

    return scene, camera, output


if __name__ == '__main__':
    # Usage: blender -b -P Depth_Renderer.py [ShapeNet directory] [model list] [output directory] [num scans per model]
    model_dir = sys.argv[-4]
    list_path = sys.argv[-3]
    output_dir = sys.argv[-2]
    num_scans = int(sys.argv[-1])

    '''Generate Intrinsic Camera Matrix'''
    # High Resolution: width = 1600,
    # Middle Resolution: width = 1600//4,
    # Coarse Resolution: width = 1600//10,

    width = 1600//4
    height = 1200//4
    focal = 1000//4
    scene, camera, output = setup_blender(width, height, focal)
    # offset is the center of images, the unit of focal here is the pixels(on the image)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    # os.system('rm -rf %s' % output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(list_path)) as file:
        model_list = [line.strip() for line in file]
    open(os.path.join(output_dir, 'blender.log'), 'w+').close()
    np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), intrinsics, '%f')
    # camera-referenced system

    num_total_f = len(model_list)

    start = time.time()
    '''rendering from the mesh to 2.5D depth images'''
    for idx, model_id in enumerate(model_list):
        # start = time.time()
        exr_dir = os.path.join(output_dir, 'exr', model_id)
        pose_dir = os.path.join(output_dir, 'pose', model_id)
        os.makedirs(exr_dir)
        os.makedirs(pose_dir)
        # os.removedirs(exr_dir)
        # os.removedirs(pose_dir)

        # Redirect output to log file
        old_os_out = os.dup(1)
        os.close(1)
        os.open(os.path.join(output_dir, 'blender.log'), os.O_WRONLY)

        # Import mesh model
        # model_path = os.path.join(model_dir, model_id, 'models/model_normalized.obj')
        # bpy.ops.import_scene.obj(filepath=model_path)

        model_path = os.path.join(model_dir, model_id + '.obj')
        bpy.ops.import_scene.obj(filepath=model_path)

        # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
        bpy.ops.transform.rotate(value=-np.pi / 2, axis=(1, 0, 0))

        # Render
        for i in range(num_scans):
            scene.frame_set(i)
            pose = random_pose()
            camera.matrix_world = mathutils.Matrix(pose)
            # output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
            output.file_slots[0].path = exr_dir + '_#.exr'
            bpy.ops.render.render(write_still=True)
            # np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')
            np.savetxt(pose_dir + '_%d.txt' % i, pose, '%f')

        # Clean up
        bpy.ops.object.delete()
        for m in bpy.data.meshes:
            bpy.data.meshes.remove(m)
        for m in bpy.data.materials:
            m.user_clear()
            bpy.data.materials.remove(m)

        # Print used time
        os.close(1)
        os.dup(old_os_out)
        os.close(old_os_out)
        print('%d/%d: %s done, time=%.4f sec' % (idx + 1, num_total_f, model_id, time.time() - start))
        os.removedirs(exr_dir)
        os.removedirs(pose_dir)
