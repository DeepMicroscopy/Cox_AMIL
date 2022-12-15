import argparse
from email.policy import default
import numpy as np
import openslide
import os
import pandas as pd
import re
import torchvision.transforms as T

from PIL import Image
from SlideRunner_dataAccess.database import Database
from tqdm import tqdm 

import sys
sys.path.append('..')

from tma_utils.tma_utils import extract_core, core_2_vips



# set random transforms
random_transform = T.Compose(
            [
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
                T.RandomApply([T.RandomRotation(180)], p=0.5),
                T.RandomApply([T.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5)
            ]
    )


def main(args):
    
    # create result directory to store single cores if necessary
    assert args.result_dir, 'result_dir must be provided'
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    # load database
    print('Loading databases ...', end=' ')
    assert args.database, 'database file must be provided!'
    DB = Database().open(args.database)

    # join annoId with classes 
    get_cores = 'SELECT Classes.uid, Classes.name, Annotations_label.annoId FROM Classes LEFT JOIN Annotations_label ON Classes.uid=Annotations_label.class'
    cores = pd.DataFrame(DB.execute(get_cores).fetchall(), columns=['core_id', 'core_label', 'anno_id'])

    # get coordinates
    get_coords = 'SELECT annoId, coordinateX, coordinateY, Annotations_coordinates.slide FROM Annotations_coordinates LEFT JOIN Annotations ON Annotations_coordinates.annoId=Annotations.uid WHERE Annotations.deleted=0'
    coords = pd.DataFrame(DB.execute(get_coords), columns=['anno_id', 'x', 'y', 'slide_id'])

    # join cores and coords
    df = pd.merge(cores, coords, how='inner', on='anno_id')

    # get slides
    get_slides = 'SELECT uid, filename, directory FROM Slides'
    slides = pd.DataFrame(DB.execute(get_slides), columns=['slide_id', 'filename', 'dir'])

    # ============================================ #
    #TODO: Change location and filenames to remove these manual steps!!!

    # clean dirs 
    slides.dir = [s.replace('\\', '/') for s in slides.dir]

    # change dirs
    orig_dirs = [s.split('/')[-1] for s in slides.dir]
    new_dirs = [str(args.data_dir) if di == '01' else os.path.join(args.data_dir, 'Delivery2') for di in orig_dirs]
    slides = slides.assign(dir=new_dirs)

    # add level 
    level = [1 if i == 5 else 0 for i in slides.slide_id]
    slides = slides.assign(level=level)

    # add patient_id
    patient_ids = [int(re.split('P|_', s)[1]) for s in df.core_label]
    df = df.assign(patient_id=patient_ids)
    # # ============================================ #
    print('Done!')

    # loop over patients (patient_id)
    for patient_id in tqdm(df.patient_id.unique(), desc='Creating WSIs'):
        # filter data from all slides and cores
        patient_df = df[df.patient_id == patient_id]
        heights = []
        widths = []
        # loop over each slide per patient
        for slide_id in sorted(patient_df.slide_id.unique()):
            slide_df = patient_df[patient_df.slide_id == slide_id]
            slide_downf = slides.level[slides.slide_id == slide_id].item() + 1.
            # loop over each core per slide and collect width and height 
            for core_id in sorted(slide_df.core_id.unique()):
                core_df = slide_df[slide_df.core_id == core_id]
                heights += [int(np.ptp(core_df.y) / slide_downf)]
                widths += [int(np.ptp(core_df.x) / slide_downf)]   
        if args.augmented:
            # add augmented versions 
            for aug_id in range(args.num_augs):
                # create new wsi 
                wsi_height = max(heights)
                wsi_width = sum(widths)
                wsi = np.zeros((wsi_height, wsi_width, 4))
                x = 0
                y = 0
                # loop over each slide again to load image into memory
                for slide_id in sorted(patient_df.slide_id.unique()):
                    slide_df = patient_df[patient_df.slide_id == slide_id]
                    slide_fn = slides.dir[slides.slide_id == slide_id].item() + '/' + slides.filename[slides.slide_id == slide_id].item()
                    slide_level = slides.level[slides.slide_id == slide_id].item()
                    slide = openslide.open_slide(str(slide_fn))
                    # loop over each core per slide to extract it 
                    for core_id in sorted(slide_df.core_id.unique()):
                        core = extract_core(slide, slide_df, core_id=core_id, level=slide_level)
                        img_core = Image.fromarray(core)
                        # perform augmentation
                        aug_core = np.array(random_transform(img_core)) 
                        height, width, _ = aug_core.shape       
                        wsi[y:height, x:(x+width)] = aug_core
                        x += width
                # convert to pyvips image and save as pyramdial .tif file 
                wsi = wsi.astype(np.uint8)
                vi = core_2_vips(wsi)
                name = 'slide_' + str(patient_id).zfill(3) + '_' + str(aug_id).zfill(2)
                vi.tiffsave(os.path.join(args.result_dir, name + '.tif'), 
                            compression='none', 
                            tile=True, 
                            tile_width=128,   # vips default size
                            tile_height=128, 
                            pyramid=True, 
                            bigtiff=True, 
                            properties=True)  
        else:
            # create new wsi 
            wsi_height = max(heights)
            wsi_width = sum(widths)
            wsi = np.zeros((wsi_height, wsi_width, 4))
            x = 0
            y = 0
            # loop over each slide again to load image into memory
            for slide_id in sorted(patient_df.slide_id.unique()):
                slide_df = patient_df[patient_df.slide_id == slide_id]
                slide_fn = slides.dir[slides.slide_id == slide_id].item() + '/' + slides.filename[slides.slide_id == slide_id].item()
                slide_level = slides.level[slides.slide_id == slide_id].item()
                slide = openslide.open_slide(str(slide_fn))
                # loop over each core per slide to extract it 
                for core_id in sorted(slide_df.core_id.unique()):
                    core = extract_core(slide, slide_df, core_id=core_id, level=slide_level)
                    height, width, _ = core.shape       
                    wsi[y:height, x:(x+width)] = core
                    x += width
            # convert to pyvips image and save as pyramdial .tif file 
            wsi = wsi.astype(np.uint8)
            vi = core_2_vips(wsi)
            name = 'slide_' + str(patient_id).zfill(3)
            vi.tiffsave(os.path.join(args.result_dir, name + '.tif'), 
                        compression='none', 
                        tile=True, 
                        tile_width=128,   # vips default size
                        tile_height=128, 
                        pyramid=True, 
                        bigtiff=True, 
                        properties=True)  


def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for extracing, augmenting and stitching cores from TMA.')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory.')
    parser.add_argument('--result_dir', type=str, default=None, help='Results directory.')
    parser.add_argument('--database', type=str, default=None, help='Database filepath.')
    parser.add_argument('--augmented', action='store_true', help='Create augmented versions of WSI.')
    parser.add_argument('--num_augs', type=int, default=10, help='Number of augmented WSIs to create.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Finished!')
    print('End of script.')