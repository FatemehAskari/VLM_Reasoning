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
log_file = os.path.join(log_folder, 'render_images_binding_{}.log'.format(timestamp))
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
parser.add_argument('--num_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.2, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.2, type=float,
    help="Minimum distance between objects along cardinal directions.")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="Minimum visible pixels per object in rendered images.")
parser.add_argument('--max_retries', default=500, type=int,
    help="Number of attempts to place an object before restarting.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="Starting index for numbering rendered images.")
parser.add_argument('--num_images', default=2, type=int,
    help="Number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="Prefix for rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for rendering.")
parser.add_argument('--output_image_dir', default='../output/images3D/',
    help="Directory for output images.")
parser.add_argument('--output_scene_dir', default='../output/scenes3D/',
    help="Directory for output JSON scene structures.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file with all scene information")
parser.add_argument('--version', default='1.0',
    help="Version string for the generated JSON file")
parser.add_argument('--license', default="Creative Commons Attribution (CC-BY 4.0)",
    help="License string for the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="Date string for the generated JSON file")
parser.add_argument('--num_features', default=10, type=int,
    help="Number of unique features for binding tasks")
parser.add_argument('--override_output_dir', default=False, action='store_true',
    help="Whether to override the output directory")

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
        output_image_dir = args.output_image_dir.replace("images", "binding/images")
        output_scene_dir = args.output_scene_dir.replace("scenes", "binding/scenes")
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
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {
            name: [round(float(c) / 255.0, 2) for c in rgb] + [1.0]
            for name, rgb in properties['colors'].items()
        }
        shape_names = list(properties['shapes'].keys())
        color_names = list(color_name_to_rgba.keys())

    for n_objects in [args.num_objects]:
        for triplet_target in range(5, 21, 5):
            base_dir = os.path.join('images', 'n_objects={}'.format(n_objects), 'triplet={}'.format(triplet_target))
            os.makedirs(base_dir, exist_ok=True)
            for i in range(30):
                trial_features = []
                for _ in range(n_objects):
                    shape = random.choice(shape_names)
                    color = random.choice(color_names)
                    trial_features.append((shape, color))
                index = i + 100 * n_objects + triplet_target * 10000
                img_path = os.path.join(base_dir, '{}.png'.format(index))
                scene_path = os.path.join(base_dir, '{}.json'.format(index))
                all_scene_paths.append(scene_path)
                print('Rendering image {}/30 for n_objects={}, triplet={}'.format(i+1, n_objects, triplet_target))
                render_scene(
                    args,
                    num_objects=n_objects,
                    output_index=index,
                    output_split=args.split,
                    output_image=img_path,
                    output_scene=scene_path,
                    objects_features=trial_features,
                    target_triplets=triplet_target
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
    os.makedirs(os.path.dirname(args.output_scene_file), exist_ok=True)
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f, indent=2)

def render_scene(args, num_objects=5, output_index=0, output_split='none', output_image='render.png', output_scene='render_json', objects_features=None, target_triplets=None):
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
        'target_triplets': target_triplets
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
    objects, blender_objects = add_binding_objects_with_triplets(scene_struct, num_objects, args, camera, objects_features, target_triplets)
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print('Rendering error: {}'.format(e))
    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

def count_triplet(object_features, n_objects):
    count1 = 0
    count2 = 0
    app = []
    for i in range(len(object_features)):
        for j in range(i+1, len(object_features)):
            for k in range(j+1, len(object_features)):
                if object_features[i]['shape'] == object_features[j]['shape'] and (
                    object_features[i]['color'] == object_features[k]['color'] or
                    object_features[j]['color'] == object_features[k]['color']
                ):
                    count1 += 1
                if check(object_features[i], object_features[j], object_features[k]) or \
                   check(object_features[i], object_features[k], object_features[j]) or \
                   check(object_features[j], object_features[k], object_features[i]):
                    app.append([object_features[i], object_features[j], object_features[k]])
                    count2 += 1
    return count1, count2, app

def check(obj1, obj2, obj3):
    if obj1['shape'] == obj2['shape'] and obj1['color'] != obj2['color'] and (
        (obj1['color'] == obj3['color'] and obj1['shape'] != obj3['shape']) or
        (obj2['color'] == obj3['color'] and obj2['shape'] != obj3['shape'])
    ):
        return True
    return False

def add_binding_objects_with_triplets(scene_struct, num_objects, args, camera, objects_features=None, target_triplets=None):
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {
            name: [round(float(c) / 255.0, 2) for c in rgb] + [1.0]
            for name, rgb in properties['colors'].items()
        }
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_map_dict = properties['shapes']
        size_mapping = list(properties['sizes'].items())
    shape_names = list(object_map_dict.keys())
    color_names = list(color_name_to_rgba.keys())
    def is_triplet(obj1, obj2, obj3):
        return (
            obj1['shape'] == obj2['shape'] and obj1['color'] != obj2['color'] and (
                (obj1['color'] == obj3['color'] and obj1['shape'] != obj3['shape']) or
                (obj2['color'] == obj3['color'] and obj2['shape'] != obj3['shape'])
            )
        )
    def count_triplets(objs):
        count = 0
        app = []
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                for k in range(j + 1, len(objs)):
                    if is_triplet(objs[i], objs[j], objs[k]) or \
                       is_triplet(objs[i], objs[k], objs[j]) or \
                       is_triplet(objs[j], objs[k], objs[i]):
                        app.append([objs[i], objs[j], objs[k]])
                        count += 1
        return count, app
    object_features = []
    count2 = 0
    app = []
    while count2 != target_triplets or len(object_features) != num_objects:
        if len(object_features) > num_objects:
            count2 = 0
            object_features = []
        shape_name = random.choice(shape_names)
        color_name = random.choice(color_names)
        obj = {'shape': shape_name, 'color': color_name}
        object_features.append(obj)
        count2, app = count_triplets(object_features)
        if count2 > target_triplets:
            object_features.pop()
            count2, app = count_triplets(object_features)
    positions = []
    objects = []
    blender_objects = []
    print("%%%", app)
    for i in range(num_objects):
        shape_name = object_features[i]['shape']
        color_name = object_features[i]['color']
        size_name, radius = random.choice(size_mapping)
        while True:
            r = random.uniform(0, 4.5)
            alpha = random.uniform(0, 2 * math.pi)
            x, y = r * math.cos(alpha), r * math.sin(alpha)
            dists_ok = all(
                math.sqrt((x - xx) ** 2 + (y - yy) ** 2) - radius - rr >= args.min_dist
                for (xx, yy, rr) in positions
            )
            margins_ok = True
            for (xx, yy, rr) in positions:
                dx = x - xx
                dy = y - yy
                for dir_name in ['left', 'right', 'front', 'behind']:
                    dir_vec = scene_struct['directions'][dir_name]
                    margin = dx * dir_vec[0] + dy * dir_vec[1]
                    if 0 < margin < args.margin:
                        margins_ok = False
                        break
                if not margins_ok:
                    break
            if dists_ok and margins_ok:
                break
        blend_obj_name = object_map_dict[shape_name]
        rgba = color_name_to_rgba[color_name]
        theta = 360.0 * random.random()
        add_object(args.shape_dir, blend_obj_name, radius, (x, y), theta=theta)
        obj_bl = bpy.context.object
        blender_objects.append(obj_bl)
        positions.append((x, y, radius))
        mat_name, mat_name_out = random.choice(material_mapping)
        add_material(mat_name, Color=rgba, logger=logger)
        pixel_coords = get_camera_coords(camera, obj_bl.location)
        objects.append({
            'shape': shape_name,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj_bl.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })
    print("&&&&&", app)
    if not check_visibility(blender_objects, args.min_pixels_per_object):
        logger.info("Some objects are occluded; retrying...")
        for obj in blender_objects:
            delete_object(obj)
        return add_binding_objects_with_triplets(scene_struct, num_objects, args, camera, None, target_triplets)
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
        print('blender --background --python binding.py -- [args]')
        print('You can also run as a standalone python script to view all arguments:')
        print('python binding.py --help')