### les images publiées sur le site scanvan doivent être floutées à certains endroits : 
    # le masque cachant la caméra et le toit de la voiture 
    # /media/scanvan/mask/20200219-151940_dhlab-car/mask.png
    # les masques de segmentation comprenant des people ou des vehicles
    # /media/scanvan/model/camera_40008603-40009302/%date/%folder/point-cloud-segmentation/scanvan-sbvp-single/2d-segmentation

from glob import glob
import imageio
import os
import sys
sys.path.append('/home/descombe/point-cloud-segmentation/spherical-images/src_scanvan/')
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tqdm
import click

@click.command()
@click.option('--folder', help="Directory of the project, containing 2d segmentation and jpg images")
def make_masks(folder: str):
    
    masks_folder = folder + "/anonymization/segmentation/"

    images = glob(folder + "/images/*.jpg")
    export_path = folder + "/anonymization/masks/"
    
    if not os.path.exists(export_path):
        os.makedirs(export_path, exist_ok=True)

    dhlab_mask_path = '/media/scanvan/mask/20200219-151940_dhlab-car/mask.png'
    dhlab_mask = imageio.imread(dhlab_mask_path)
    
    for image in tqdm.tqdm(images):
        
        filename = os.path.basename(image)
        probmap_people = cv.imread(masks_folder + filename.replace(".jpg","-probmap-people.png"), 0)
        ret,mask_people = cv.threshold(probmap_people,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        kernel = np.array([[0, 0, -1, 0, 0],[0, 0, -1, 0, 0],[0, 0, 1, 0, 0],[0, 0, 1, 0, 0],[0, 0, 1, 0, 0]], dtype=np.uint8)
        mask_p_filter_it1 = cv.filter2D(mask_people, -1, kernel)
        mask_p_filter = cv.filter2D(mask_p_filter_it1, -1, kernel)

        probmap_cars = cv.imread(masks_folder + filename.replace(".jpg","-probmap-vehicle.png"), 0)
        ret,mask_cars = cv.threshold(probmap_cars,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        mask_thresh = cv.bitwise_or(mask_p_filter,mask_cars)
        mask_inv = cv.bitwise_not(mask_thresh)

        # resize at the same dimensions as the original image (and dhlab-car mask), add dhlab-car mask
        mask_resized = np.zeros_like(dhlab_mask)
        height_scaled = dhlab_mask.shape[0] / mask_inv.shape[0]
        width_scaled = dhlab_mask.shape[1] / mask_inv.shape[1]

        for i in range(mask_inv.shape[0]):
            for j in range(mask_inv.shape[1]):
                mask_resized[ int( np.floor( i * width_scaled ) ) : int( np.floor( ( i + 1 ) * width_scaled ) ) , int( np.floor( j * height_scaled ) ): int( np.floor( ( j + 1 ) * height_scaled ) ) ] += mask_inv[i,j]

        #mask_total = cv.bitwise_and(dhlab_mask,mask_resized)
        image_mask = export_path + filename.replace(".jpg", "-mask.png")
        cv.imwrite(image_mask, mask_resized)



if __name__ == '__main__':
    make_masks()