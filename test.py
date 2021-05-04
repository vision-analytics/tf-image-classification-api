import argparse
import base64
import json
import requests
import sys
import time

import cv2
import numpy as np

from modules import utilities

WS_URL_LOCAL="http://localhost:5000/api/v1/classify_image"

def run(source: str, path: str):
    global WS_URL_LOCAL

    data = {} #request body
    
    if source == 'file':        
        # read image from url
        image = utilities.read_image_from_file(path) 
        
        if image is not None:
            # convert image to base64
            base64_str = utilities.convert_image_to_base64(image)

            # prepare request body
            data = {"img": base64_str}
    elif source == 'url':
        data = {"url": path}
    else:
        print(f"invalid source {source}")
        return

    start_time = time.time()

    # send post request
    r = requests.post(WS_URL_LOCAL, json=data)

    print("inference time {} seconds".format(time.time()-start_time))

    print(f"status_code: {r.status_code}, response: {r.text}")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, choices=['url', 'file'], help='source type')
    parser.add_argument("-p", "--path", required=True, help='full path/url')
    args = parser.parse_args()

    run(source=args.source, path=args.path)