import numpy as np
import openslide
import pandas as pd
import pyvips
import re
from typing import List, Tuple


def extract_core(slide: openslide.OpenSlide, 
                 df: pd.DataFrame, 
                 level: int=0,
                 core_id: int=None, 
                 core_label: str=None) -> np.ndarray:
    """Extracts core as numpy array from TMA slide.

    Args:
        slide (openslide.OpenSlide): TMA slide.
        df (pd.DataFrame): DataFrame with core annotations.
        level (int, optional): Level to sample. Defaults to 0.
        core_id (int, optional): Core identifier (e.g. 123). Defaults to None.
        core_label (str, optional): Core label (e.g. P123_1). Defaults to None.

    Raises:
        ValueError: If core_id and core_label do not match when both are provided.
        ValueError: If neither core_id nor core_label are provided.

    Returns:
        np.ndarray: Core as numpy array with RGBA format.
    """

    if core_label is not None:
        XY = df[['core_id', 'core_label', 'x', 'y']].loc[df['core_label'] == core_label]
    elif core_id is not None:
        XY = df[['core_id', 'core_label', 'x', 'y']].loc[df['core_id'] == core_id]
    elif core_label is not None and core_id is not None:
        XY = df[['core_id', 'core_label', 'x', 'y']].loc[df['core_id'] == core_id]
        if XY.core_label.unique()[0] != core_label:
            raise ValueError('core_id and core_label do not match') 
    else:
        raise ValueError('Expected either core_id or core_label.')

    if XY.shape[0] == 0:
        print(f'No annotations for core_id {core_id} available.')
        
    else:    
        x_min, y_min = int(min(XY.x)), int(min(XY.y))
        width, height = int(np.ptp(XY.x)), int(np.ptp(XY.y))
        
        downf = slide.level_downsamples[level]
        location = x_min, y_min
        size = int(width/downf), int(height/downf)   
        core = slide.read_region(location=location, size=size, level=level)

        return np.asarray(core) 



def core_2_vips(core: np.ndarray) -> pyvips.Image:
    """Converts numpy array to pyvips image.

    Args:
        core (np.ndarray): Numpy array.

    Returns:
        pyvips.Image: Pyvips image. 
    """
    height, width, bands = core.shape
    vi = pyvips.Image.new_from_memory(core, width, height, bands, 'uchar')
    return vi 



def check_patientID_range(slide_id: int, low: int, high: int, df: pd.DataFrame) -> None:
    """Checks whether all patient ID's on a TMA are in the correct range. Verifys that no other labels are on the TMA.

    Args:
        slide_id (int): Slide identifier.
        low (int): Lowest patient ID.
        high (int): Highest patient ID.
        df (pd.DataFrame): Dataframe with core annotations.

    Returns:
        bool: True if all ID's are in the correct range.
    """
    persons_str = df.core_label.loc[df.slide_id == slide_id]
    persons_set = set([int(re.split('P|_', i)[1]) for i in persons_str])
    correct = (min(persons_set)>=low) and (max(persons_set)<=high)
    print(f'slide_id {slide_id}: {correct}')



def check_coreID_range(slide_id: int, df: pd.DataFrame, min: int = 1500, max: int = 6000, verbose: bool=False) -> Tuple[List[np.ndarray]]:
    """Checks the size of the core annotations. Cores that are too large indicate annotation mistakes (e.g. 2 cores apart with same label).

    Args:
        slide_id (int): Slide identifier
        df (pd.DataFrame): Dataframe with core annotations.
        min (int, optional): Minimum core size. Defaults to 1500.
        max (int, optional): Maximum core size. Defaults to 6000.
    """
    x_ranges, y_ranges = [], []
    too_big, too_small = [], []
    subdf = df[df.slide_id == slide_id]
    for core_id in subdf.core_id.unique():
        XY = subdf[['x', 'y']].loc[subdf['core_id'] == core_id]
        if XY.shape[0] > 0:
            x_range, y_range = np.ptp(XY.x), np.ptp(XY.y)
            x_ranges.append(x_range)
            y_ranges.append(y_range)
            if x_range > max or y_range > max:
                too_big.append((core_id, x_range, y_range)) 
            if x_range < min or y_range < min:
                too_small.append((core_id, x_range, y_range)) 
    
    if verbose:
        if len(too_big) > 0:
            print(f'\nSlide {slide_id} with {len(too_big)} suspicously large core annotations.')
            for core_id, x_range, y_range in too_big:
                print(f'Core {core_id}, x_range {x_range}, y_range {y_range}')
        else:
            print(f'\nSlide {slide_id} without suspicously large core annotations.')
        
        if len(too_small) > 0:
            print(f'\nSlide {slide_id} with {len(too_small)} suspicously small core annotations.')
            for core_id, x_range, y_range in too_small:
                print(f'Core {core_id}, x_range {x_range}, y_range {y_range}')
        else:
            print(f'\nSlide {slide_id} without suspicously small core annotations.')

    return x_ranges, y_ranges, too_big, too_small


    

    
    


    

    


