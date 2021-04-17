from glob import glob
import os
import numpy as np
import cv2 as cv
import click
import tqdm

@click.command()
@click.option('--folder', help='folder of images to be blurred')
def gaussian_blur(folder: str):

#    model_name = max(folder.split('/'), key=len).replace('-normalized','')
#    folder_scanvan = [x for x in glob('/media/scanvan/record/camera_40008603-40009302/*%s' % model_name.split('loop')[-1] ) if x.split('/')[-1][:8] == model_name[16:24]][0]
    folder_out = os.path.join(folder, "gaussian_blur")
    os.mkdir(folder_out)
    mask_inv = cv.imread('/media/scanvan/mask/20200219-151940_dhlab-car/mask.png',0)
    mask = cv.bitwise_not(mask_inv)
    
#    images = glob(folder + "/images/*.jpg")
    images = glob(folder + "/blur/*.png")
    for img in tqdm.tqdm(images):
        
        filename = os.path.basename(img).split('.')[0]
#        bmp_img = folder_scanvan + "/" + filename + ".png"
        bmp_img = img
        bmp = cv.imread(bmp_img)
        blur = cv.blur(bmp,(20,20),0)
        out = bmp.copy()
        out[mask!=0] = blur[mask!=0]

        cv.imwrite(os.path.join(folder_out, filename + ".png"), out)
    
if __name__ == '__main__':
    gaussian_blur()
