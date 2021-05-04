# -*- coding: utf-8 -*-
#! /usr/bin/env python

import os
import base64
import configparser
import uuid
import pickle
import requests
from io import BytesIO

import cv2
import imutils
import numpy as np
from PIL import Image, ImageOps

config = None

def create_transaction_id():
    """Function to create unique transaction id
    
    Args: 
        None
    
    Returns: 
        str. unique id

    """
    return str(uuid.uuid1())

def convert_image_to_base64(frame):
    """Function to convert single image to base64 (encoder)
    
    Args: 
        frame: numpy.ndarray. BGR image
    
    Returns: 
        base64 string

    """
    print("converting image to base64..")
    
    retval, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def convert_base64_to_image(b64_string):
    """Function to convert base64 to image (decoder)
    
    Args: 
        b64_string: string. base64 string
    
    Returns: 
        numpy.ndarray. BGR image

    """
    print("converting base64 to image..")
    nparr = None
    npimg = None

    try:
        nparr = np.fromstring(base64.b64decode(b64_string), np.uint8)
        npimg = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    except Exception as e:
        print(f"unable to convert base64 (image).. {e}")
        
    return npimg

def read_image_from_file(file_path):
    """Function to read image from file
    
    Args: 
        file_path: str. full path of image
    
    Returns: 
        numpy.ndarray. BGR image

    """
    #print("reading image from file..")
    try:
        img = cv2.imread(file_path)
    except:
        print("invalid path/image")
        return 
    finally:
        if not isinstance(img, (np.ndarray) ):
            print("invalid path/image")
            return 
    return img

def read_image_from_url(url):
    """Function to read image from url
    
    Args: 
        url: str.
    
    Returns: 
        numpy.ndarray. BGR image

    """
    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        print("invalid url/image")
        return 
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_image(image, width=None, height=None):
    """Function to resize image
    
    Args: 
        image: numpy.ndarray.
        width: int. 
        height: int. 
    
    Returns: 
        image: numpy.ndarray. resized image 

    """
    #print("resizing image..")
    assert(width is not None or height is not None)
    
    if width is not None and height is not None:
        return cv2.resize(image, (width, height))
    if width is not None:
        return imutils.resize(image, width=width)
    elif height is not None:
        return imutils.resize(image, height=height)

def read_config_file(config_file_path="config.ini"):
    global config
    
    print("reading configuration from file.. {}".format(config_file_path))
    config = configparser.ConfigParser()

    # check if log file exists or not
    if not os.path.isfile(config_file_path):
        return "<unable to read config_file for given path: {}>".format(config_file_path)
    
    # read config from file
    config.read(config_file_path)

def get_config(section_name=None, key=None, key_type="str"):
    """Function to read configuration from file

    Args: 
        section_name: str. (optional)
        key: str. (optional)

    Returns: 
        *. config value
    """
    global config
    if config is None:
        read_config_file(config_file_path=os.environ["CONFIG_PATH"])

    # check section
    if section_name is None:
        return "<please provide section name for config>"
    if section_name not in config:
        return "<invalid section name({})>".format(section_name)
    
    # read section 
    section = config._sections[section_name]

    # if key not is provided return all section else return key value only
    if key is None:
        return section
    else:
        if key not in section:
            return "<invalid key name({}) for section: {}>".format(key, section_name)
        else:
            if key_type == "bool":
                return config.getboolean(section_name, key)
            else:
                return section[key]
