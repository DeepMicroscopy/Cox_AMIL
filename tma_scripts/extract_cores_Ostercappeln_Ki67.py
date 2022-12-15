import argparse
import numpy as np
import openslide
import os
import pandas as pd
import re

from SlideRunner_dataAccess.database import Database
from tqdm import tqdm 

import sys
sys.path.append('..')

from tma_utils.tma_utils import extract_core, core_2_vips

# misc params
MICRONS_PER_MM = 1000
SIZE_THRESH = 1000


def main(args):
    
    # create result directory to store single cores if necessary
    assert args.result_dir, 'result_dir must be provided'
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

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
    # change directories
    dirs = [str(args.data_dir)] * 2 + [os.path.join(args.data_dir, 'Delivery3')] * 5
    slides = slides.assign(dir=dirs)

    # add level 
    level = [1] * 2 + [0] * 5
    slides = slides.assign(level=level)

    # add patient_id
    patient_ids = [int(re.split('P|_', s)[1]) for s in df.core_label]
    df = df.assign(patient_id=patient_ids)
    # ============================================ #
    print('Done!')

    # extract and save cores
    for slide_id in tqdm(df.slide_id.unique(), desc='Processing slide'):

        slide_dir, slide_fn = slides[['dir', 'filename']][slides.slide_id == slide_id].values[0]

        slide = openslide.open_slide(os.path.join(slide_dir, slide_fn))
        mpp_x = float(slide.properties['openslide.mpp-x'])
        mpp_y = float(slide.properties['openslide.mpp-y'])
        
        slide_df = df[df.slide_id == slide_id]
        slide_level = slides.level[slides.slide_id == slide_id].item()

        # loop over all cores in subdf
        for idx, core_id in enumerate(tqdm(slide_df.core_id.unique(), desc='Extracting cores')):
            # extract core and save as .tif
            core = extract_core(slide, slide_df, level=slide_level, core_id=core_id)
            if core is not None:
                height, width, _ = core.shape
                # omit small cores
                if height < SIZE_THRESH or width < SIZE_THRESH:
                    continue
                else:
                    vi = core_2_vips(core)
                    name = str(slide_id) + '_' + df.core_label.loc[df.core_id == core_id].unique().item()
                    vi.tiffsave(os.path.join(args.result_dir, name + '.tif'), 
                                compression='none', 
                                tile=True, 
                                tile_width=128,   # vips default size
                                tile_height=128, 
                                pyramid=True, 
                                bigtiff=True, 
                                xres=1/mpp_x*MICRONS_PER_MM,  # vips default px per mm
                                yres=1/mpp_y*MICRONS_PER_MM,
                                properties=True) 
    

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for extracing single cores from TMA.')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory.')
    parser.add_argument('--result_dir', type=str, default=None, help='Results directory.')
    parser.add_argument('--database', type=str, default=None, help='Database filepath.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Finished!')
    print('End of script.')




    