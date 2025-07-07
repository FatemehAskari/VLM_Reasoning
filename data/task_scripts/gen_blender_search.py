from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
timestamp = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_folder, 'render_images_search_{}.log'.format(timestamp))
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='blender_utils/base_scene.blend',
    help="Base blender file on which all scenes are based; includes ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='blender_utils/properties.json',
    help="JSON file defining objects, materials, sizes, and colors.")
parser.add_argument('--shape_dir', default='blender_utils/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='blender_utils/materials',
    help="Directory where .blend files for materials are stored")

# Settings for objects
parser.add_argument('--num_objects', default=20, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.2, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.1, type=float,
    help="Minimum distance between objects along cardinal directions.")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="Minimum visible pixels per object in rendered images.")
parser.add_argument('--max_retries', default=500, type=int,
    help="Number of attempts to place an object before restarting.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="Starting index for numbering rendered images.")
parser.add_argument('--num_images', default=5, type=int,
    help="Number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="Prefix for rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for rendering.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="Directory for output images.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="Directory for output JSON scene structures.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file with all scene information")
parser.add_argument('--version', default='1.0',
    help="Version string for the generated JSON file")
parser.add_argument('--license', default="Creative Commons Attribution (CC-BY 4.0)",
    help="License string for the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="Date string for the generated JSON file")
parser.add_argument('--override_output_dir', default=False, action='store_true',
    help="Whether to override the output directory")
parser.add_argument('--task', default="search", type=str, choices=['search', 'search_counterfactual'],
    help="The task to render images for: search or search_counterfactual.")

# Rendering options
parser.add_argument('--width', default=640, type=int,
    help="Width of rendered images (pixels)")
parser.add_argument('--height', default=480, type=int,
    help="Height of rendered images (pixels)")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="Random jitter for key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="Random jitter for fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="Random jitter for back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="Random jitter for camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="Number of samples for rendering.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="Minimum number of bounces for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="Maximum number of bounces for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="Tile size for rendering.")

def main(args):
    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    if args.override_output_dir:
        output_image_dir = args.output_image_dir.replace("images", args.task + "/images")
        output_scene_dir = args.output_scene_dir.replace("scenes", args.task + "/scenes")
    else:
        output_image_dir = args.output_image_dir
        output_scene_dir = args.output_scene_dir
    img_template = os.path.join(output_image_dir, img_template)
    scene_template = os.path.join(output_scene_dir, scene_template)

    if not os.path.isdir(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.isdir(output_scene_dir):
        os.makedirs(output_scene_dir)
    
    all_scene_paths = []
    num_images = args.num_images
    num_objects = args.num_objects

    for i in range(num_images):
        logger.info('\nRendering image %d / %d' % (i, num_images))
        index = i + 100 * num_objects + args.start_idx
        img_path = img_template % (index)
        logger.info('\nImage path: %s, index: %d' % (img_path, index))
        scene_path = scene_template % (index)
        all_scene_paths.append(scene_path)
        logger.info('\nRendering scene %d with %d objects' % (index, num_objects))
        render_scene(args,
            num_objects=num_objects,
            output_index=index,
            output_split=args.split,
            output_image=img_path,
            output_scene=scene_path
        )

    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)

def render_scene(args, num_objects=5, output_index=0, output_split='none', output_image='render.png', output_scene='render_json'):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    load_materials(args.material_dir)
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object
    def rand(L):
        return 2.0 * L * (random.random() - 0.5)
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()
    delete_object(plane)
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)
    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)
    logger.info('\nObjects added')
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    logger.info('\nStarting rendering')
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            logger.info('\nRendering error %s' % e)
            print(e)
    logger.info('\nRendering done')
    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

def add_random_objects(scene_struct, num_objects, args, camera):
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [np.round(float(c) / 255.0, 2) for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())
    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        max_retries = args.max_retries + 10*i
        logger.info('\nAdding object %d of %d\n' % (i, num_objects))
        size_name, r = random.choice(size_mapping)
        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > max_retries:
                logger.info("Max retries exceeded: %d" % max_retries)
                for obj in blender_objects:
                    delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)
            radius = random.uniform(0, 4.5)
            alpha = random.uniform(0, 2 * math.pi)
            x, y = radius * math.cos(alpha), radius * math.sin(alpha)
            logger.info('\nTrying to place object at %f, %f, within %f' % (x, y, radius))
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(margin, args.margin, direction_name)
                        print('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break
            if dists_good and margins_good:
                break
        if args.task == 'search':
            if i == 0:
                obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
                color_name, rgba = "red", dict(color_name_to_rgba)["red"]
            elif i % 2 == 0:
                obj_name, obj_name_out = "SmoothCube_v2", dict(object_mapping)["SmoothCube_v2"]
                color_name, rgba = "red", dict(color_name_to_rgba)["red"]
            else:
                obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
                color_name, rgba = "green", dict(color_name_to_rgba)["green"]
        else:  # search_counterfactual
            if i % 2 == 0:
                obj_name, obj_name_out = "SmoothCube_v2", dict(object_mapping)["SmoothCube_v2"]
                color_name, rgba = "red", dict(color_name_to_rgba)["red"]
            else:
                obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
                color_name, rgba = "green", dict(color_name_to_rgba)["green"]
        logger.info('\nAdding object: %s, %s, %s' % (obj_name_out, color_name, rgba))
        if obj_name == 'SmoothCube_v2':
            r *= 0.9
        theta = 360.0 * random.random()
        add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        mat_name, mat_name_out = random.choice(material_mapping)
        add_material(mat_name, Color=rgba, logger=logger)
        pixel_coords = get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
    if not all_visible:
        print('Some objects are occluded; replacing objects')
        logger.info('\nSome objects are occluded; replacing objects')
        for obj in blender_objects:
            delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
    return objects, blender_objects

def compute_all_relationships(scene_struct, eps=0.2):
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships

def check_visibility(blender_objects, min_pixels_per_object):
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i+1], p[i+2], p[i+3]) for i in range(0, len(p), 4))
    logger.info('\nColor count: %s' % color_count)
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True

def render_shadeless(blender_objects, path='flat.png'):
    render_args = bpy.context.scene.render
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False
    set_layer(bpy.data.objects['Lamp_Key'], 2)
    set_layer(bpy.data.objects['Lamp_Fill'], 2)
    set_layer(bpy.data.objects['Lamp_Back'], 2)
    set_layer(bpy.data.objects['Ground'], 2)
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat
    bpy.ops.render.render(write_still=True)
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat
    set_layer(bpy.data.objects['Lamp_Key'], 0)
    set_layer(bpy.data.objects['Lamp_Fill'], 0)
    set_layer(bpy.data.objects['Lamp_Back'], 0)
    set_layer(bpy.data.objects['Ground'], 0)
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing
    return object_colors

def extract_args(input_argv=None):
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv

def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))

def delete_object(obj):
    for o in bpy.data.objects:
        o.select = False
    obj.select = True
    bpy.ops.object.delete()

def get_camera_coords(cam, pos):
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)

def set_layer(obj, layer_idx):
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)

def add_object(object_dir, name, scale, loc, theta=0):
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1
    name_path = name
    filepath = os.path.join(object_dir, '%s.blend' % name_path)
    with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
        name = [name for name in data_from.objects][0]
    filename = os.path.join(object_dir, '%s.blend' % name_path, 'Object', name)
    bpy.ops.wm.append(filename=filename)
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name
    x, y = loc
    bpy.context.scene.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))

def load_materials(material_dir):
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)

def add_material(name, **properties):
    obj = bpy.context.active_object
    for i in range(len(obj.data.materials)):
        obj.data.materials.pop()
    mat_count = len(bpy.data.materials)
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count
    obj.data.materials.append(mat)
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]
    logger = properties.get('logger', None)
    for inp in group_node.inputs:
        if inp.name in properties:
            logger.info("color input: %s" % properties[inp.name])
            inp.default_value = properties[inp.name]
    mat.node_tree.links.new(group_node.outputs['Shader'], output_node.inputs['Surface'])

if __name__ == '__main__':
    if INSIDE_BLENDER:
        argv = extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print('blender --background --python search.py -- [args]')
        print('You can also run as a standalone python script to view all arguments:')
        print('python search.py --help')