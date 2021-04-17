#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from dh_segment import inference, post_processing
import tensorflow as tf
import numpy as np
import os
import cv2
import json
from imageio import imsave
from typing import List
import click
from tqdm import tqdm
from utils import LabelCodes
from glob import glob
import yaml

def binarize_doors(probability_map,
                   kernel_size):
    """
    Binarization function for 'door' class
    """
    probs_door = post_processing.cleaning_probs(probability_map, sigma=0)
    binary_fixed_door = post_processing.thresholding(probs_door, threshold=0.2)
    binary_otsu_door = post_processing.thresholding(probs_door, threshold=-1)
    binary_door = binary_fixed_door * binary_otsu_door
    binary_door = post_processing.cleaning_binary(binary_door, kernel_size=kernel_size)
    return binary_door

def polygonize(binary_mask,
               scaling,
               min_area=0,
               approx=0):
    """
    Create a polygon from a binary blob
    """
    polygons = post_processing.find_polygonal_regions(binary_mask, min_area=min_area)
    if polygons:
        polygons = [np.array(p * scaling, np.int) for p in polygons]
        if approx > 0:
            polygons = [cv2.approxPolyDP(p, approx, True) for p in polygons]

        return [p.tolist() for p in polygons]
    else:
        return None

def find_image_path(main_path: str):
    config = open(main_path + "/config.yaml")
    parsed_config = yaml.load(config, Loader=yaml.FullLoader)
    image_path = parsed_config['frontend']['image']
    
    return image_path

@click.command()
@click.option('--model_dir', help="Directory of the model's weights")
@click.option('--model_name', help="Name of the segmentation model (for creating respective directory)")
@click.option('--export_dir', help="Directory to the main project folder")
def segment(model_dir: str,
            model_name: str,
            export_dir: str):
    """
    Given the filenames of the images, segment it using the model provided and save the resulting grayscale
    images into ``export_dir``
    """

    # initialize architecture 
    export_dir_pcs = export_dir + "/point-cloud-segmentation"
    if not os.path.exists(export_dir_pcs):
        os.makedirs(export_dir_pcs, exist_ok=True)
        
    export_dir_pcs_segmodel = export_dir_pcs + "/%s" % model_name
    if not os.path.exists(export_dir_pcs_segmodel):
        os.makedirs(export_dir_pcs_segmodel, exist_ok=True)
    else:
        print(f"directory {export_dir_pcs_segmodel} already exists.")
        
    image_path = find_image_path(export_dir)
    img_jpg_dir = export_dir_pcs_segmodel + "/img_jpg/"
    if not os.path.exists(img_jpg_dir):
        os.makedirs(img_jpg_dir, exist_ok=True)
        print('--- converting images...---')
        os.system('./bmp_to_jpg %s %s' % (image_path, img_jpg_dir))

    images_filenames = glob('%s/*' % img_jpg_dir)

    export_dir_images = export_dir_pcs_segmodel + "/2d-segmentation/"
    if not os.path.exists(export_dir_images):
        os.makedirs(export_dir_images, exist_ok=True)     

    with tf.Session():
        model = inference.LoadedModel(model_dir)

        for img_to_pred in tqdm(images_filenames, total=len(images_filenames)):
            predictions = model.predict(img_to_pred)

            probability_maps = predictions['probs'][0]
            original_shape = predictions['original_shape']
            ratio_scaling = (original_shape[0] / probability_maps.shape[0],
                             original_shape[1] / probability_maps.shape[1])

            binary_sky = binarize_doors(probability_maps[:, :, 0], 20)
            binary_building = binarize_doors(probability_maps[:, :, 1], 20)
            binary_vehicle = binarize_doors(probability_maps[:, :, 2], 20)
            binary_people = binarize_doors(probability_maps[:, :, 3], 20)

            polygon_sky = polygonize(binary_sky, ratio_scaling, approx=10)
            polygon_building = polygonize(binary_building, ratio_scaling, approx=10)
            polygon_vehicle = polygonize(binary_vehicle, ratio_scaling, approx=10)
            polygon_people = polygonize(binary_people, ratio_scaling, approx=10)

            polygons_dict = {'sky': polygon_sky,
                             'building': polygon_building,
                            'vehicle': polygon_vehicle,
                            'people': polygon_people}

            img_basename = os.path.basename(img_to_pred).split('.')[0]

            # save polygons coordinates
            polygons_file = os.path.join(export_dir_images, f'{img_basename}-polygons.json')
            with open(polygons_file, 'w') as file:
                json.dump(polygons_dict, file)

            # save probability maps
            sky_probmap_filename = os.path.join(export_dir_images, f'{img_basename}-probmap-sky.png')
            building_probmap_filename = os.path.join(export_dir_images, f'{img_basename}-probmap-building.png')
            vehicle_probmap_filename = os.path.join(export_dir_images, f'{img_basename}-probmap-vehicle.png')
            people_probmap_filename = os.path.join(export_dir_images, f'{img_basename}-probmap-people.png')
            imsave(sky_probmap_filename, np.uint8(probability_maps[:, :, 0] * 255))
            imsave(building_probmap_filename, np.uint8(probability_maps[:, :, 1] * 255))
            imsave(vehicle_probmap_filename, np.uint8(probability_maps[:, :, 2] * 255))
            imsave(people_probmap_filename, np.uint8(probability_maps[:, :, 3] * 255))

            # create and save label image
            label_image_filename = os.path.join(export_dir_images, f'{img_basename}-labels.png')
            labels_image = np.zeros_like(probability_maps[:, :, 0])
            labels_image[binary_sky > 0] = LabelCodes.sky
            labels_image[binary_building > 0] = LabelCodes.building
            labels_image[binary_vehicle > 0] = LabelCodes.vehicle
            labels_image[binary_people > 0] = LabelCodes.people
            imsave(label_image_filename, np.uint8(labels_image))


if __name__ == '__main__':
    segment()
