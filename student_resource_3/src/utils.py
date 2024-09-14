import re
import constants
import os
import requests
import pandas as pd
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image
import json

def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s==None or str(s)=='nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit



def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

def download_image(row, save_folder, retries=3, delay=3):
    image_link = row['image_link']
    entity_name = row['entity_name']
    entity_name = f"given this image just return me the {entity_name} of the item.Just return the numerical value along with dimension and please ensure that units/dimensions are always mentioned. If voltage/wattage is asked return the unit as volt/wattage ."
    entity_value = row['entity_value']

    if not isinstance(image_link, str):
        return filename, None, None

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return filename, entity_name, entity_value

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return filename, entity_name, entity_value
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            time.sleep(delay)

    create_placeholder_image(image_save_path)  # Create a black placeholder image for invalid links/images
    return filename, None, None

def download_images(df, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_status = {}

    if allow_multiprocessing:
        download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(download_image_partial, df.to_dict('records')), total=len(df)))

        for result in results:
            if result[0] is not None:
                download_status[result[0]] = [result[1], result[2]]
    else:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename, entity_name, entity_value = download_image(row, save_folder=download_folder, retries=3, delay=3)
            download_status[filename] = [entity_name, entity_value]

    # Save the download status dictionary as a JSON file
    with open('download_status_train.json', 'w') as f:
        json.dump(download_status, f, indent=2)

    return download_status

if __name__ == "__main__":
    df = pd.read_csv("/66e31d6ee96cd_student_resource_3/student_resource_3/dataset/train.csv")
    # links = df["image_link"].tolist()
    download_status = download_images(df, "/training")
    print(download_status)
    print("Download completed. Status saved in download_status.json")