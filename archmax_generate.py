import math
import random
from pathlib import Path
from zipfile import ZipFile

import numpy
import bpy
from math import radians
import os
import argparse
import datetime
import sys
import subprocess
import pkg_resources
import shutil
import typing
import mathutils
import copy


def install_packages(pkgs: set):
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = pkgs - installed
    if missing:
        print('Missing packages: {} \nInstall? (y/n)'.format(missing))
        if input() == 'y':
            print('Installing required packaged:', missing)
            python_ex = sys.executable
            subprocess.check_call([python_ex, '-m', 'ensurepip'])
            subprocess.check_call([python_ex, '-m', 'pip', 'install', *missing])

        else:
            print('Quitting, please install manually')
            exit(0)


install_packages({'tqdm', 'numpy', 'pillow', 'owlready2', 'pandas', 'sympy', 'huggingface-hub'})

from huggingface_hub import hf_hub_download
import pandas as pd
import sympy
from sympy import *
from owlready2 import *
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import owlready2

# Constants
MAIN_CLASSES = ['Residential', 'Commercial', 'Industrial']
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

labels_template = {
    'id': 0,

    'Residential': 0,  # main labels
    'Commercial': 0,
    'Industrial': 0,

    'Cafe': 0,
    'Hotel': 0,
    'Restaurant': 0,
    'Store': 0,
    'MiscCommercial': 0,
    'Suburban': 0,
    'MiscResidential': 0,
    'CountryHouse': 0,
    'ConstructionSite': 0,
    'MiscIndustrial': 0,
    'PowerPlant': 0,
    'WaterTreatment': 0,

    'Door': -1,
    'Window': -1,
    'Awning': -1,
    'Billboard': -1,
    'Porch': -1,
    'Sign': -1,
    'Table': -1,
    'TiledRoof': -1,
    'TiledRoofTop': 0,
    'TiledRoofBottom': 0,
    'VendingMachine': -1,
    'WallSign': -1,
    'Statue': -1,
    'Chimney': -1,
    'Pipe': -1,
    'Machine': -1,
    'Truck': -1,
    'Car': -1,

    'DoorPos': -1,  # Model variants and positions
    'DoorModel': -1,

    'WindowModel': [[-1] * 7],
    'nWindows': 0,
    'WindowPos': [[0] * 7],

    'AwningModel': [[-1] * 7],
    'nAwnings': 0,
    'AwningPos': [[0] * 7],

    'BillboardModel': -1,
    'BillboardPos': -1,

    'PorchModel': -1,
    'PorchPos': -1,

    'SignModel': -1,
    'SignPos': -1,

    'TableModel': -1,
    'TablePos': -1,

    'TiledRoofBottomModel': -1,
    'TiledRoofBottomPos': -1,

    'TiledRoofTopModel': -1,
    'TiledRoofTopPos': -1,

    'VendingMachineModel': -1,
    'VendingMachinePos': -1,

    'WallSignModel': -1,
    'WallSignPos': -1,

    'StatueModel': -1,
    'StatuePos': -1,

    'ChimneyModel': -1,
    'ChimneyPos': -1,

    'PipeModel': -1,
    'PipePos': -1,

    'MachineModel': -1,
    'MachinePos': -1,

    'TruckModel': -1,
    'TruckPos': -1,

    'CarModel': -1,
    'CarPos': -1,

    'CameraPoint': [(0, 0, 0)],  # Camera
    'CameraPos': [(40.6, 71.2, 17.7)],

    'WallTextures': [[9] * 2],  # Textures
    'RimTextures': [[0] * 4],
    'FloorTexture': 4,

    'BackgroundModel': -1,

    'Contrast': 1,
    'Gamma': 1
}
primitive_labels_template = {
    'Door': -1,
    'Window': -1,
    'Awning': -1,
    'Billboard': -1,
    'Porch': -1,
    'Sign': -1,
    'Table': -1,
    'TiledRoof': -1,
    'VendingMachine': -1,
    'WallSign': -1,
    'Statue': -1,
    'Chimney': -1,
    'Pipe': -1,
    'Machine': -1,
    'Truck': -1,
    'Car': -1,
}

# complexity_levels = {
#     1: np.array([MODELS]),
#     2: np.array([MODELS, TEXTURES]),
#     3: np.array([MODELS, TEXTURES, CAM_SMALL]),
#     4: np.array([MODELS, TEXTURES, CAM_SMALL, BACKGROUND]),
#     5: np.array([MODELS, TEXTURES, BACKGROUND, CAM_LARGE]),
#     6: np.array([MODELS, TEXTURES, BACKGROUND, CAM_LARGE, CONTRAST, GAMMA]),
# }

owl_and = owlready2.class_construct.And
owl_or = owlready2.class_construct.Or
owl_not = owlready2.class_construct.Not
owl_rest = owlready2.class_construct.Restriction

N_ALTS = 5  # number of alternative models for each primitive feature


# CLASSES
class Position:
    def __init__(self, loc, rot):
        self.loc = loc
        self.rot = rot


class Model:
    def __init__(self, ob, pos: Position):
        self.ob = ob
        self.pos = pos
        self.defaultPos = pos

    def move(self, new_pos):
        self.ob.location = new_pos.loc
        self.rotate(new_pos.rot)
        self.pos = new_pos

    def rotate(self, angle):
        self.ob.rotation_mode = 'XYZ'
        c_rot = self.ob.rotation_euler
        angle = angle - self.pos.rot
        self.ob.hide_set(False)
        self.ob.rotation_euler[2] += radians(angle)
        self.ob.hide_set(True)

        # bpy.ops.object.select_all(action='DESELECT')  # deselect all

        # self.ob.select_set(state=True)
        # bpy.ops.transform.rotate(value=math.radians(angle), orient_axis=axis)
        # self.ob.select_set(state=False)

    def hide(self):
        self.ob.hide_render = True

    def show(self):
        self.ob.hide_render = False

    def reset(self):
        if self.defaultPos is not None:
            self.move(self.defaultPos)


class StdoutMute:
    def __init__(self):
        self.old = None
        self.logfile = None
        self.on = False

    def turn_on(self):
        # redirect output
        if not self.on:
            self.on = True
            self.old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            self.logfile = tempfile.TemporaryFile(mode='w')

    def turn_off(self):
        # disable output redirection
        if self.on:
            self.on = False
            self.logfile.close()
            os.dup(self.old)
            os.close(self.old)


class OwlClass:
    def __init__(self, onto_class: owlready2.ThingClass, labels_template: dict):
        self.name = onto_class.name
        self.onto_class = onto_class
        self.labels_template = copy.deepcopy(labels_template)
        self.disjoints = self.__init_disjoints()
        self.choices = self.__init_choices__()

    def generate_labels(self) -> dict:
        choice = sample(self.choices)  # this is an and clause
        labels = copy.deepcopy(self.labels_template)
        for s in choice.args:  # set obligatory labels
            if isinstance(s, sympy.Symbol):
                labels[s.name] = 1
            elif isinstance(s, sympy.Not):
                if isinstance(s.args[0], sympy.Symbol):
                    labels[s.args[0].name] = 0

        for key in labels:  # set dcs
            if labels[key] == -1:
                labels[key] = 1 if myRand() else 0

        return labels

    def eval(self, labels: dict):
        keys = list(labels.keys())
        subs_dict = {}
        for k in keys:
            subs_dict[symbols(k)] = labels[k] == 1
        for choice in self.choices:
            if choice.subs(subs_dict):
                return True
        return False

    def __init_choices__(self):
        class_expression = np.array(self.onto_class.equivalent_to)
        for disj_class in self.disjoints:
            class_expression = np.append(class_expression, np.array([owl_not(e) for e in disj_class.equivalent_to]))

        for my_class in self.get_all_superclasses(self.onto_class):
            for superclass in my_class.is_a:
                if isinstance(superclass, owlready2.ClassConstruct):
                    class_expression = np.append(class_expression, [superclass])
                else:
                    class_expression = np.append(class_expression, list(superclass.equivalent_to))

        class_expression = owl_and(class_expression)
        class_expression = OwlClass.__convert_to_sympy__(class_expression)

        class_expression = simplify_logic(class_expression, form='dnf',
                                          deep=True, force=True)  # disjunction of conjunctions, aka choices

        if not class_expression:
            raise Exception('Ontology Error: Class ' + self.name + ' is unsatisfiable')

        return np.array(class_expression.args) if isinstance(class_expression, sympy.Or) else np.array(
            [class_expression])

    def __init_disjoints(self):
        out = set([])
        for superclass in self.get_all_superclasses(self.onto_class):
            for disj_list in superclass.disjoints():
                for entity in disj_list.entities:
                    if entity.name != superclass.name and entity.name != 'Feature':
                        out = out.union(self.get_hierarchy(entity))

        out = np.array(list(out))
        return out

    def get_hierarchy(self, onto_class):
        out = np.array([onto_class])
        for subclass in list(onto_class.subclasses()):
            out = np.append(out, self.get_hierarchy(subclass))
        return out

    def get_all_superclasses(self, onto_class):
        out = np.array([onto_class])
        for superclass in onto_class.is_a:
            if isinstance(superclass, owlready2.ThingClass) and superclass != owl.Thing:
                out = np.append(out, self.get_all_superclasses(superclass))
        return out

    @staticmethod
    def __convert_to_sympy__(expr: typing.Union[owl_and, owl_or, owl_not, owl_rest]):
        expr_type = type(expr)
        # AND CONSTRUCT
        if expr_type == owl_and:
            out = OwlClass.__convert_to_sympy__(expr.Classes[0])
            for e in expr.Classes[1:]:
                out = out & OwlClass.__convert_to_sympy__(e)
            return out

        # OR CONSTRUCT
        elif expr_type == owl_or:
            out = OwlClass.__convert_to_sympy__(expr.Classes[0])
            for e in expr.Classes[1:]:
                out = out | OwlClass.__convert_to_sympy__(e)
            return out

        # NOT CONSTRUCT
        elif expr_type == owl_not:
            return ~OwlClass.__convert_to_sympy__(expr.Class)

        # SYMBOL
        elif expr_type == owl_rest:
            name = expr.__getattr__('value').name
            return symbols(name)


# POSITIONS

# walls
wall_pos = np.array([Position((6.75, -3.19, 4.85), 0),
                     Position((6.75, 3.21, 4.85), 0),
                     Position((2.68, 6.93, 4.96), 90),
                     Position((-2.91, 6.93, 4.96), 90),
                     Position((3.46, -3.19, 13.32), 0),  # 2nd floor
                     Position((3.46, 3.96, 13.32), 0),
                     Position((-1.98, 6.85, 13.32), 90)])

# surroundings
surround_pos = np.array([Position((12.87, -12.11, 11.28), 0),
                         Position((12.87, 18.01, 11.28), 0),
                         Position((-8.59, 18.01, 11.28), 0)])

# porches
porch_pos = np.array([Position((9.87, 0.38, 2.56), 0),
                      Position((-0.14, 10.14, 2.56), 90)])

# wall signs
wall_sign_pos = np.array([Position((8.02, -5.93, 5.87), 0),
                          Position((6.31, 8.20, 5.87), 90),
                          Position((-6.24, 8.2, 5.87), 90),
                          Position((4.72, -5.93, 13.94), 0),  # 2nd floor
                          Position((2.87, 8.07, 13.94), 90),
                          Position((-6.57, 8.07, 13.94), 90)])

# billboards
billboard_pos = np.array([Position((7.11, 0.2, 10.57), 0),
                          Position((0.4, 7.34, 10.57), 90),
                          Position((4.32, 0.2, 17.46), 0)])

# tile tops
tile_top_pos = np.array([Position((-1.77, 0.33, 20.99), 0)])

# tile bottoms
tile_bottom_pos = np.array([Position((6.33, 0.17, 10.56), 0)])

# statues
statue_pos = np.array([Position((0, 0, 24.65), 0)])


# Setup

def keep_images_check(arg: str):
    if arg.isdecimal():
        return abs(int(arg))
    else:
        if arg.lower() == 'all':
            return arg.lower()
        else:
            raise argparse.ArgumentTypeError('must be an integer or "all"')


def positive_int(arg: str):
    if arg.isdecimal():
        return abs(int(arg))
    else:
        raise argparse.ArgumentTypeError('must be an integer')


MODELS, TEXTURES, CAM_SMALL, BACKGROUND, CAM_LARGE, CONTRAST, GAMMA = 0, 1, 2, 3, 4, 5, 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help='Size of desired dataset', required=True)

    parser.add_argument('--textures', '-t', default=False, action='store_true')
    parser.add_argument('--cam', default='none', choices=['none', 'small', 'large'])
    parser.add_argument('--background', '-bg', default=False, action='store_true')
    parser.add_argument('--contrast', '-c', default=False, action='store_true')
    parser.add_argument('--gamma', '-g', default=False, action='store_true')

    parser.add_argument('--ontology_path', type=str, default=os.path.join(ROOT_DIR, 'ontology.owl'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Directory to store the generated assets')
    parser.add_argument('--resolution', type=int, nargs=2, default=(224, 224),
                        help='Number of pixels in the side of the rendered square images')
    parser.add_argument('--keep', default='all', type=keep_images_check,
                        help='Number of images to keep in jpg format, use "all" to keep all')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable high verbosity')
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]
    return parser.parse_args(argv)


def prepare():
    # parse command-line arguments
    my_args = parse_args()

    # write config file
    my_args.output_dir = os.path.join(my_args.output_dir, datetime.datetime.now().strftime('%d%m_%H%M%S'))
    os.makedirs(my_args.output_dir)
    with open(os.path.join(my_args.output_dir, 'config.txt'), 'w') as f:
        f.write(str(my_args))
        f.close()

    # verbosity
    if not my_args.verbose:
        my_args.mute = StdoutMute()
        my_args.mute.turn_on()

    # open blender project
    proj_path = os.path.join(ROOT_DIR, 'blender_project')
    if not os.path.isdir(proj_path):
        my_args.mute.turn_off()
        print('Blender project (12.8gb) needed, download? (y/n)')
        if input() == 'y':
            images_zip = hf_hub_download(repo_type='dataset', repo_id="bruno-cotrim/arch-max-blender-proj",
                                         filename="blender_project.zip")

            with ZipFile(images_zip) as f:
                ZipFile.extractall(f, path=ROOT_DIR)
        else:
            print('Quitting ...')
            exit(0)
        my_args.mute.turn_on()

    bpy.ops.wm.open_mainfile(filepath=os.path.join(proj_path, 'models.blend'))

    # set render settings
    obs = bpy.data.objects
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.resolution_x = tuple(my_args.resolution)[0]
    scene.render.resolution_y = tuple(my_args.resolution)[1]
    my_args.scene = scene

    # create image folder
    my_args.image_dir = os.path.join(my_args.output_dir, 'images')
    os.makedirs(my_args.image_dir, exist_ok=False)



    # get camera
    my_args.camera = my_args.scene.camera
    if not my_args.verbose:
        my_args.mute.turn_off()
    # assets
    my_args.assets = get_assets()
    # ontology
    my_args.onto = get_ontology(os.path.join('file://', my_args.ontology_path)).load()
    my_args.classes = init_owl_classes(my_args.onto)
    return my_args


# Asset handling
def init_owl_classes(onto: owlready2.Ontology):
    classes = {}
    for c in list(onto.Building.subclasses()):
        subclass_list = []
        for sub_class in list(c.subclasses()):
            curr_class = OwlClass(sub_class, primitive_labels_template)
            subclass_list.append(curr_class)
        classes[c.name] = subclass_list
    return classes


def wrangle_obs(src: dict, name: str, n: int, copy_n=0):
    col = []
    if copy_n > 0:
        for mesh_id in range(n):
            copy_col = [src[name + str(mesh_id)]]
            for copy_id in range(1, copy_n):
                copy_col.append(src[name + str(mesh_id) + '.00' + str(copy_id)])
            col.append(copy_col)
    else:
        for mesh_id in range(n):
            col.append(src[name + str(mesh_id)])
    return np.array(col)


def init_models(col: numpy.ndarray, pos_list: np.ndarray):
    if col.ndim > 1:
        for arr_ix, arr in enumerate(col):
            col[arr_ix] = [Model(ob, pos_list[ix]) for ix, ob in enumerate(arr)]
    else:
        for ob_ix, ob in enumerate(col):
            col[ob_ix] = Model(ob, pos_list[0])
    return col


def get_assets():
    obs = bpy.data.objects
    mats = bpy.data.materials
    assets = {
        'Window': init_models(wrangle_obs(obs, 'window', 5, 7), wall_pos),
        'Awning': init_models(wrangle_obs(obs, 'awning', 5, 7), wall_pos),
        'Door': init_models(wrangle_obs(obs, 'door', 5), wall_pos),
        'Porch': init_models(wrangle_obs(obs, 'porch', 5), porch_pos),
        'Sign': init_models(wrangle_obs(obs, 'sign', 5), surround_pos),
        'Table': init_models(wrangle_obs(obs, 'table', 5), surround_pos),
        'WallSign': init_models(wrangle_obs(obs, 'wallsign', 5), wall_sign_pos),
        'VendingMachine': init_models(wrangle_obs(obs, 'vmachine', 5), wall_pos),
        'TiledRoofTop': init_models(wrangle_obs(obs, 'tiletop', 3), tile_top_pos),
        'TiledRoofBottom': init_models(wrangle_obs(obs, 'tilebottom', 3), tile_bottom_pos),
        'Billboard': init_models(wrangle_obs(obs, 'billboard', 5), billboard_pos),
        'Statue': init_models(wrangle_obs(obs, 'statue', 5), statue_pos),
        'Chimney': init_models(wrangle_obs(obs, 'chimney', 5), statue_pos),
        'Pipe': init_models(wrangle_obs(obs, 'pipe', 5), wall_pos),
        'Machine': init_models(wrangle_obs(obs, 'machine', 5), surround_pos),
        'Truck': init_models(wrangle_obs(obs, 'truck', 5), porch_pos),
        'Car': init_models(wrangle_obs(obs, 'car', 5), porch_pos),

        'wall': wrangle_obs(obs, 'wall', 2),
        'wall_mat': wrangle_obs(mats, '._wall', 10),
        'rim': wrangle_obs(obs, 'rim', 4),
        'rim_mat': wrangle_obs(mats, '._rim', 3),
        'floor': obs['floor'],
        'floor_mat': wrangle_obs(mats, '._floor', 5),
        'background': init_models(wrangle_obs(obs, 'bg', 5), np.array([None])),
        'camera': obs['camera'],
    }
    return assets


def resetRender():
    # args.camera.location = (45.90, 80.63, 18.11)
    # args.camera.location = (0, 0, 0)
    assets = args.assets
    for k in assets.keys():
        col = np.array(assets[k])
        col = col.flatten()
        if isinstance(col[0], Model):
            for m in col:
                m.reset()
                m.hide()


# Randomization
def randomize_textures(labels: dict):
    assets = args.assets
    for id, wall in enumerate(assets['wall']):
        mat_ix = sample(len(assets['wall_mat']))
        mat = assets['wall_mat'][mat_ix]
        labels['WallTextures'][0][id] = mat_ix
        if wall.data.materials:
            # assign to 1st material slot
            wall.data.materials[0] = mat
        else:
            # no slots
            wall.data.materials.append(mat)

    for id, rim in enumerate(assets['rim']):
        mat_ix = sample(len(assets['rim_mat']))
        mat = assets['rim_mat'][mat_ix]
        labels['RimTextures'][0][id] = mat_ix
        if rim.data.materials:
            # assign to 1st material slot
            rim.data.materials[0] = mat
        else:
            # no slots
            rim.data.materials.append(mat)

    floor_mat_ix = sample(len(assets['floor_mat']))
    labels['FloorTexture'] = floor_mat_ix
    assets['floor'].material_slots[0].material = assets['floor_mat'][floor_mat_ix]
    return labels


def randomize_camera(large, labels: dict):
    camera = args.camera
    if large:
        dist_bounds = [60, 120]
        theta_bounds = [10, 80]
        phi_bounds = [15, 80]
        focal_var = 15
    else:
        dist_bounds = [60, 80]
        theta_bounds = [35, 55]
        phi_bounds = [40, 70]
        focal_var = 5

    distance = sample(np.arange(dist_bounds[0], dist_bounds[1]))
    location = sample_sphere(distance, theta_bounds, phi_bounds)
    camera.location = location
    focus_point = mathutils.Vector((sample(focal_var), sample(focal_var), sample(focal_var)))
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    labels['CameraPos'] = [(location[0], location[1], location[2])]
    labels['CameraPoint'] = [(focus_point.x, focus_point.y, focus_point.z)]
    return labels
    # Use * instead of @ for Blender <2.8


def cart_to_spher(v):  # assumes x,y,z > 0
    x, y, z = v[0], v[1], v[2]
    phi = math.acos((z / math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))))
    theta = math.atan(y / x)
    return math.degrees(theta), math.degrees(phi)


def sample_sphere(radius, theta_bounds, phi_bounds):
    theta, phi = -1, -1
    v = np.array([0.0, 0.0, 0.0])
    while not (theta_bounds[0] <= theta <= theta_bounds[1]) or not (phi_bounds[0] <= phi <= phi_bounds[1]):
        rng = np.random.default_rng()
        rng.standard_normal(3, out=v)
        v = np.abs(v)
        v = v / np.linalg.norm(v)
        v = v * radius
        theta, phi = cart_to_spher(v)
    return v

    # distance = sample(np.arange(60, 150))
    # focus_point = mathutils.Vector((5, 0, 0))
    # looking_direction = camera.location - focus_point
    # rot_quat = looking_direction.to_track_quat('Z', 'Y')
    #
    # camera.rotation_euler = rot_quat.to_euler()
    # # Use * instead of @ for Blender <2.8
    # camera.location = rot_quat @ mathutils.Vector((0, 0, distance))


# Dataset generation
def place(models, positions, available):
    pos = sample(available)
    np.delete(available, pos)
    model = sample(models)
    model.show()
    model.move(positions[pos])
    return available


def sample(array: typing.Union[int, np.ndarray], count=1, return_array=False):
    if type(array) == np.ndarray and array.ndim > 1:
        res = array[np.random.choice(len(array), size=count, replace=False)]
    else:
        res = np.random.choice(array, size=count, replace=False)

    if count == 1 and not return_array:
        return res[0]
    else:
        return res


def myRand(odds=50):
    return sample(100) < odds


def set_labels(main_label: str):
    # MAIN LABELS
    labels = copy.deepcopy(labels_template)
    labels[main_label] = 1

    # PRIMITIVE AND SUBCLASS LABELS
    subclass = sample(args.classes[main_label])  # random subclass from main class
    labels.update(subclass.generate_labels())

    # TILED ROOF
    if labels['TiledRoof'] == 1:
        labels['TiledRoofBottom'] = 1
        if labels['Statue'] == 0:
            labels['TiledRoofTop'] = 1

    for subcl in args.classes[main_label]:
        labels[subcl.name] = 1 if subcl.eval(labels) else 0

    return labels
    # MISC LABELS

    # RANDOMIZATION


def add_noise(labels):
    if args.textures:
        labels = randomize_textures(labels)
    if args.cam == 'small':
        labels = randomize_camera(False, labels)
    elif args.cam == 'large':
        labels = randomize_camera(True, labels)
    if args.background:
        model = sample(len(args.assets['background']))
        labels['BackgroundModel'] = model
        args.assets['background'][model].show()
    return labels


def generateDataset():

    assets = args.assets
    resetRender()

    labels_df = pd.DataFrame(columns=list(labels_template.keys()))
    render_time = datetime.timedelta(0)
    if not args.verbose:
        args.mute.turn_on()

    for i in tqdm(range(args.n)):

        labels = set_labels(list(args.classes)[i % 3])
        # Complexity

        labels = add_noise(labels)


        # Special cases
        wall_free_ix = set(range(len(wall_pos)))
        if labels['Door'] == 1:
            model = sample(N_ALTS)  # decide
            pos_ix = sample(4)
            labels['DoorModel'] = model  # label
            labels['DoorPos'] = pos_ix
            door = assets['Door'][model]  # execute
            door.show()
            door.move(wall_pos[pos_ix])
            wall_free_ix = wall_free_ix.difference({pos_ix})

        if labels['Window'] == 1:
            n_windows = sample(len(wall_free_ix)) + 1
            labels['nWindows'] = n_windows
            w_pos = sample(np.array(list(wall_free_ix)), count=n_windows, return_array=True)
            wall_free_ix = wall_free_ix.difference(set(w_pos))
            for pos_ix in w_pos:
                model = sample(N_ALTS)
                labels['WindowPos'][0][pos_ix] = 1
                labels['WindowModel'][0][pos_ix] = model
                assets['Window'][model][pos_ix].show()

        if labels['Awning'] == 1:
            aw_possible_pos = set(range(len(wall_pos))).difference(wall_free_ix)
            n_awnings = sample(len(aw_possible_pos)) + 1
            labels['nAwnings'] = n_awnings
            a_pos = sample(np.array(list(aw_possible_pos)), count=n_awnings, return_array=True)
            for pos_ix in a_pos:
                model = sample(N_ALTS)
                labels['AwningPos'][0][pos_ix] = 1
                labels['AwningModel'][0][pos_ix] = model
                assets['Awning'][model][pos_ix].show()

        # Models that share space - must check availability
        vm_pos = set(range(4)).difference({labels['DoorPos']})
        vm_pos = wall_pos[list(vm_pos)]

        for lbl_list, pos_list in zip(
                [['Sign', 'Table', 'Machine'], ['Porch', 'Car', 'Truck'], ['VendingMachine', 'Pipe']],
                [surround_pos, porch_pos, vm_pos]):
            free_ix = set(range(len(pos_list)))
            for lbl in lbl_list:
                if labels[lbl] == 1:
                    pos_ix = sample(np.array(list(free_ix)))
                    free_ix = free_ix.difference({pos_ix})
                    model = sample(N_ALTS)
                    labels[lbl + 'Pos'] = pos_ix
                    labels[lbl + 'Model'] = model
                    assets[lbl][model].show()
                    assets[lbl][model].move(pos_list[pos_ix])

        # Models that don't share space
        for lbl, pos_list in zip(['Billboard', 'WallSign', 'Statue', 'Chimney', 'TiledRoofTop', 'TiledRoofBottom'],
                                 [billboard_pos, wall_sign_pos, statue_pos, statue_pos, tile_top_pos, tile_bottom_pos]):
            if labels[lbl] == 1:
                pos_ix = sample(np.arange(len(pos_list)))
                model = sample(len(assets[lbl]))
                labels[lbl + 'Pos'] = pos_ix
                labels[lbl + 'Model'] = model
                assets[lbl][model].show()
                assets[lbl][model].move(pos_list[pos_ix])

        # Store labels
        labels['id'] = i
        new_row = pd.DataFrame(labels, index=[i])
        labels_df = pd.concat([labels_df, new_row])

        # RENDER

        before_render = datetime.datetime.now()
        image_path = os.path.join(args.image_dir, str(i) + '.jpg')
        args.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        elapsed = datetime.datetime.now() - before_render
        render_time = render_time + elapsed
        resetRender()

        # GAMMA AND CONSTRAST

        if args.gamma or args.contrast:
            img = Image.open(image_path)

            if args.gamma:
                filter = ImageEnhance.Brightness(img)
                value = random.random() + 0.5
                img = filter.enhance(value)
                labels['Gamma'] = value

            if args.contrast:
                filter = ImageEnhance.Contrast(img)
                value = random.random() + 0.5
                img = filter.enhance(value)
                labels['Contrast'] = value

            img.save(image_path)

        if i % 1000 == 0:
            labels_df.to_csv(os.path.join(args.output_dir, 'labels_' + str(i) + '.csv'))

    if not args.verbose:
        args.mute.turn_off()

    labels_df.to_csv(os.path.join(args.output_dir, 'labels.csv'))
    return render_time


# File handling

def save_img_as_np():
    images = np.zeros((args.n, args.resolution, args.resolution, 3), dtype='uint8')
    for i in range(args.n):
        pic = Image.open(os.path.join(args.image_dir, str(i) + ".png"))
        images[i] = np.asarray(pic.getdata(), dtype='uint8').reshape((args.resolution, args.resolution, 3))

    if args.keep != 'all':
        if args.keep == 0:
            shutil.rmtree(args.image_dir)
        else:
            for i in range(args.keep, args.n):
                os.remove(os.path.join(args.image_dir, str(i) + ".png"))

    np.save(os.path.join(args.output_dir, 'images.npy'), images)


if __name__ == '__main__':
    # DEFINITIONS
    args = prepare()
    print('Rendering ' + str(args.n) + ' images, this might take a while.')
    start = datetime.datetime.now()
    render_time = generateDataset()
    stop = datetime.datetime.now()
    print('Done! Dataset saved at ' + args.output_dir + '.\n(Total elapsed:' + str(
        (stop - start).seconds) + ' | Rendering: ' + str(render_time.seconds) + '| Overhead: ' + str(
        ((stop - start) - render_time).seconds) + ')')

    # save_img_as_np()
