#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from dataclasses import dataclass
from collections import Counter
import json
import numpy as np
from shapely.geometry import Point, mapping, shape
from typing import List, Tuple
import struct
import os
import pandas as pd
import yaml
import tqdm
import imageio
from glob import glob

@dataclass
class LabelCodes:
    background: int = 0
    people: int = 255
    vehicle: int = 196
    building: int = 128
    sky: int = 80


@dataclass
class Point3D:
    """
    Class containing point 3D info

    Attributes:
        id: the id of point given by the photogrametry pipeline
        coordinates: 3D coordinates in space
        labels: the labels given in the 2D images
        color: RGB color of the point
    """
    id: int
    coordinates: Tuple[float, float, float]
    labels: List[int]
    color: Tuple[np.uint8, np.int8, np.uint8] = (0, 0, 0)

    def assign_color(self, param):
        counter = Counter(self.labels)
        color = np.array([0, 0, 0], np.uint8)
        if param is "shared":
            unique_labels = list(counter.keys())
            counts = np.array(list(counter.values()))
            weights = counts / sum(counts)
            for lbl, w in zip(unique_labels, weights):
                if lbl == LabelCodes.people:
                    color += np.uint8([w * 255, 0, w * 255])
                elif lbl == LabelCodes.building:
                    color += np.uint8([0, w * 255, 0])
                elif lbl == LabelCodes.sky:
                    color += np.uint8([0, 0, w * 255])
                elif lbl == LabelCodes.background:
                    color += np.uint8([w * 123, w * 123, w * 123])
                elif lbl == LabelCodes.vehicle:
                    color += np.uint8([w * 255, 0, 0])
                    
        elif param is "dominant":
            lbl, _ = counter.most_common()[0]
            if lbl == LabelCodes.people:
                color += np.uint8([255, 0, 255])
            elif lbl == LabelCodes.building:
                color += np.uint8([0, 255, 0])
            elif lbl == LabelCodes.sky:
                color += np.uint8([0, 0, 255])
            elif lbl == LabelCodes.background:
                color += np.uint8([123, 123, 123])
            elif lbl == LabelCodes.vehicle:
                color += np.uint8([255, 0, 0])
 
        self.color = color    
    

class MetaInstanceInfo:
    """
    MetaInstanceInfo is a class to reunite the info concerning multiple 2D instances that in fact
    constitute one single 3D object.

    Attributes:
        id_meta_instance: the id of the MetaInstanceInfo object
        id_instances: a list of ids of the 2D instances
        labels: a list of the corresponding labels (of the 2D instances)
        points3d_ids: a list of points belonging to the same 3D object
    """
    def __init__(self,
                 id_meta_instance,
                 id_instances=None,
                 labels=None,
                 points3d_ids=None,
                 global_label=None,
                 label_confidence=None):
        self.id = id_meta_instance
        self.id_instances = id_instances if id_instances is not None else list()
        self.labels = labels if labels is not None else list()
        self.points3d_ids = points3d_ids if points3d_ids is not None else list()
        self.global_label = global_label
        self.label_confidence = label_confidence

    def compute_global_label_and_score(self):
        self.global_label, n_occurences = Counter(self.labels).most_common()[0]
        self.label_confidence = n_occurences / len(self.labels)

    def to_dict(self):
        return vars(self).copy()

    @classmethod
    def from_json(cls, json_filename):
        with open(json_filename, 'r') as f:
            json_metainstances = json.load(f)

        for j, meatainstance in enumerate(json_metainstances):
            # change key 'id'
            json_metainstances[j]['id_meta_instance'] = json_metainstances[j].pop('id')

        return [cls(**jmeta) for jmeta in json_metainstances]


class Instance2D:
    def __init__(self,
                 id_instance,
                 image_filename,
                 label,
                 points3d_ids=None,
                 points2d_coordinates=None,
                 polygon2d=None):
        """
        Instance2D corresponds to a 2D segmented instance.
        Use normalized 2d coordinates.

        Attributes:
            id_instance: id (int) of the object instance, starts at 0 for each image_filename
            image_filename: filename of the image where the instance is
            label: label of the instance
            points3d_ids: the ids of the points 3d which belong to the instance
            points2d_coordinates: coordinates of the 2d points related to the corresponding point3d.
            Has same length as points3d_ids
            polygon2d: shapely Polygon of the instance object.
        """
        self.id = id_instance
        self.image_filename = image_filename
        self.label = label
        self.points3d_ids = points3d_ids if points3d_ids is not None else list()
        self.points2d_coordinates = points2d_coordinates if points2d_coordinates is not None else list()
        self.polygon2d = polygon2d

    def is_point2d_within_instance(self, point):
        return self.polygon2d.contains(Point(point))

    def to_dict(self):
        new_dict = vars(self).copy()
        for k, v in new_dict.items():
            if k == 'polygon2d':
                new_dict[k] = mapping(v)  # use shapely.geometry.shape to convert to shapely Polygon again
        return new_dict

    @classmethod
    def from_json(cls, json_filename):
        with open(json_filename, 'r') as f:
            json_instances = json.load(f)

        for j, instance in enumerate(json_instances):
            json_instances[j]['polygon2d'] = shape(instance['polygon2d'])
            # change key 'id'
            json_instances[j]['id_instance'] = json_instances[j].pop('id')

        return [cls(**jinst) for jinst in json_instances]


def save_ply_data(ply_data: list,
                  ply_file: str,
                 file_format: str):
    
    assert file_format in ["ascii", "binary", "bin"], "Unknown export format"
    if file_format is "binary" or "bin":
        file_format = "binary_little_endian"
    
    header_str = 'ply\nformat %s 1.0\nelement vertex %d\nproperty float64 x\nproperty float64 y\nproperty float64 z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\nend_header\n' %(file_format, len(ply_data))
    
    if file_format is "ascii":
        with open(ply_file, 'w') as f:
            f.write(header_str)
            for line in ply_data:
                f.write(" ".join(map(str, line)))
                f.write("\n")

    elif file_format is "binary_little_endian":
        with open(ply_file, 'wb') as f:
            f.write(header_str.encode('ascii'))
            for x,y,z,r,g,b in ply_data:
                export = struct.pack('<dddBBB',x,y,z,r,g,b)
                f.write(export)
                   

                    
def make_sfm_data(main_dir, density):
    
    assert density in ["sparse", "dense"], "model has to be sparse or dense"
    
    # get extrinsics and views : camera positions and rotations, set an index for all images (cameras)
    sfm_data_path = main_dir + "/%s_transformation.dat" % density
    extrinsics = pd.read_table(sfm_data_path, sep = ' ', header = None)
    
    config_dict = yaml.load(open(main_dir + "/config.yaml", 'r'))
    image_path = config_dict['frontend']['image'] # rajouter "/{image_filename}.bmp"
    
    extrinsics_dict = {}
    views_dict = {}
    
    print('-- get views and extrinsics --')
    for key, row in tqdm.tqdm(extrinsics.iterrows(), total=len(extrinsics)):       
        filename = row[0]
        im = imageio.imread(image_path + "/" + filename)
        height, width = im.shape[0:2]
        translation = list(row[1:4])
        rotation = np.array(row[4:], dtype=np.float64).reshape(3,3).tolist()
        
        views_dict.update({key: {'filename': filename, 'width': width, 'height': height}})
        extrinsics_dict.update({key: {'rotation': rotation, 'center': translation}})

    """
    pour trouver les points qui sont vus dans chaque img:
    le nb de points augmente à chaque triplet -> assigner index à chaque point (chaque ligne), comparer la longueur des fichiers. 
    on arrive à trouver à quel triplet chaque point est assigné
    """
    # get structure : for each point, by which triplet it is seen
    print('-- get triplets information --')
    resections = glob(main_dir + "/%s/*structure.xyz" % density)
    triplet_length = []
    for resec in tqdm.tqdm(resections, total=len(resections)):
        pt_list = pd.read_table(resec, header = None)
        triplet_idx = int(resec.split('/')[-1].split('_')[0])
        pt_ids = len(pt_list)
        triplet_length.append([triplet_idx, pt_ids])

    print('-- get structure information --')
    structure_dict = {}
    xyz_structure = main_dir + "/%s_structure.xyz" % density
    pt_list = np.loadtxt(xyz_structure)

    for i in tqdm.tqdm(range(len(pt_list)), total=len(pt_list)):
        X = pt_list[i][0:3].tolist()
        rgb = [int(x) for x in pt_list[i][3:]]  

        if i <= triplet_length[0][1]:
            triplet = triplet_length[0][0]
        else:
            triplet_length.pop(0)
            triplet = triplet_length[0][0]

        structure_dict.update({i: {'X': X, 'color_rgb': rgb, 'triplet': triplet}})
    """
    if constraints have been exported (correspondencies between points -> viewpoints) :
    """
    if os.path.exists(main_dir + "/%s_constraint.dat" % density):
        file_in = open(main_dir + "/%s_constraint.dat" % density)
        constraint = file_in.read().split('\n')

        for i in range(len(constraint)):
            views = [int(x) for x in constraint[i].split()[7:]]
            if structure_dict.get(i) is not None:
                value = structure_dict.get(i)
                value['views'] = views
                structure_dict.update({i : value})
                
    else:
        print("Constraints have not been exported for this model")

        
    views = []
    for key, value in views_dict.items():
        views.append({"key": key, "value": value})

    extrinsics = []
    for key, value in extrinsics_dict.items():
        extrinsics.append({"key": key, "value": value})

    structure = []
    for key, value in structure_dict.items():
        structure.append({"key": key, "value": value})

    sfm_dict = {"image_path": image_path, "views": views, "extrinsics": extrinsics, "structure": structure}
    
    if not os.path.exists(main_dir + "/point-cloud-segmentation/"):
        os.makedirs(main_dir + "/point-cloud-segmentation/", exist_ok=True)

    with open(main_dir + "/point-cloud-segmentation/sfm_data_%s.json" % density, 'w') as f:
        json.dump(sfm_dict, f, indent=1)

    return image_path, structure_dict, views_dict, extrinsics_dict


def _convert_list_to_dict(data_list): # -> this exists already in point_labelling
    data_dict = dict()
    for element in data_list:
        assert element['key'] not in data_dict.keys(), "Key entry already exists"
        data_dict.update({element['key']: element['value']})
    return data_dict


def get_sfm_data_as_dict(sfm_json_filename):
    """
    Read sfm data json file, and transforms it into dictionary
    """
    with open(sfm_json_filename, 'r') as fp:
        sfm_data = json.load(fp)

    image_path = sfm_data['image_path']
    structure_dict = _convert_list_to_dict(sfm_data['structure'])
    views_dict = _convert_list_to_dict(sfm_data['views'])
    extrinsics_dict = _convert_list_to_dict(sfm_data['extrinsics'])
    
    return image_path, structure_dict, views_dict, extrinsics_dict


def views_inverse_mapping(views_dict: dict):
    """
    Create a dictionary which keys are 'filename' and values are the 'key'.
    (instead of the original key: id_view, value: filename_view)
    """

    views_reversed = dict()
    for id_view, value in views_dict.items():
        basename = value['filename']
        views_reversed[basename] = id_view

    return views_reversed


def omvs_to_omvg_idx(dict_path: str, views_dict: dict):
    """
    openMVS and openMVG images indices are not the same. 
    correspondencies of openMVS indices <-> images names are stored in a txt file at the same place as sfm_data_dense.bin
    """
    assert os.path.exists(dict_path), \
        "Correspondencies not found between openMVS and openMVG indices. \n Interrupting."
    
    dict_text = open(dict_path,'r')
    text_list = dict_text.read().split('\n')
    keys_mvs = []
    names = []
    for i in range(0, len(text_list)-1, 2):
        keys_mvs.append(int(text_list[i].split(' ')[-1]))
        names.append(text_list[i+1].split('/')[-1].split('.')[0])

    mvs_idx = dict(zip(keys_mvs,names))
    mvs_idx_reversed = dict(zip(names,keys_mvs))

    sfm_idx_reversed = views_inverse_mapping(views_dict)
    omvs_to_ovmg_dict = {}
    
    for file, omvg_id in sfm_idx_reversed.items():
        omvs_key = (mvs_idx_reversed.get(file))

        omvs_to_ovmg_dict.update({omvs_key: omvg_id})
        
    return omvs_to_ovmg_dict


def project_spherical(pt3d, img_key, extrinsics_dict):
    
    """
    simple projection on a sphere of the points of the scene:
    p' = R^{T} ( p - t ) / || p - t || 

    We obtain (x,y,z) (cartesian) coordinates and must find the corresponding spherical coordinates.
    """
    R = np.array(extrinsics_dict.get(img_key)['rotation'])    
    c = np.array(extrinsics_dict.get(img_key)['center'])

    translate = pt3d - c
    x,y,z = np.matmul( np.transpose( R ), translate ) / np.linalg.norm( translate )
    rho = np.sqrt( x * x + y * y + z * z )
    theta = np.arccos( z / rho ) # longitude = [ 0, π ]
    phi = np.arctan2( y , x ) # latitude = [ -π, π ]

    x = ( phi + np.pi ) / ( 2 * np.pi ) 
    if x < 0.5:
        x += 0.5
    else:
        x -= 0.5
    y = 1 - ( theta / np.pi )
    
    return x, y

