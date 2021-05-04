import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile

from . import utilities


class Classifier:

    def __init__(self):
        """ Generic class for classifier
    
        Attributes:
            
        """
        print("initializing classifier..")

        self.model_path = utilities.get_config(section_name="CLASSIFIER", key="model_path")
        self.img_size = int(utilities.get_config(section_name="CLASSIFIER", key="img_size"))

        self.memory_fraction = float(utilities.get_config(section_name="GPU", key="memory_fraction"))

        self.detection_graph = None
        self.session = None

        self.prepare_model()

    def preprocess_image(self, frame):
        """Function to predict document type of given image
        
        Args: 
            image: numpy.ndarray. BGR image
        
        Returns: 
            numpy.ndarray. expanded_dims (1, x, x, 3) rgb

        """
        
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype(np.float32, copy=False)
        frame -= [104, 117, 123]
        frame = np.expand_dims(frame, axis=0)

        #print(frame.shape)
        return frame

    def prepare_model(self):
        """Function to prepare model
        Args: 
        
        Returns: 
        """
      
        # limit gpu usage
        print("limit for gpu usage: %{}".format(self.memory_fraction*100))

        config = tf.ConfigProto() 
        config.gpu_options.per_process_gpu_memory_fraction = self.memory_fraction
        #config.gpu_options.visible_device_list = "0"

        self.session = tf.Session(graph = tf.Graph(), config=config)
        with self.session.graph.as_default():
            tf.keras.backend.set_session(self.session)
        print("loading classifier model...")
        try:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                self.session = tf.Session(graph=self.detection_graph)  
                
        except Exception as e:
            print(f"unable to load model for classifier! {e}")
            raise Exception(f'unable to load model for classifier! {e}')


    def predict(self, image):
        """Function to predict class of given image
        
        Args: 
            image: numpy.ndarray. BGR image
        
        Returns: 
            dict: pred_result

        """
        print("running prediction...")
        data = {}
        data["predictions"] = []
    
        processed_image = self.preprocess_image(image)

        try:
            input  = self.detection_graph.get_tensor_by_name('input:0')  
            output = self.detection_graph.get_tensor_by_name('predictions:0')
            acc_percentage = self.session.run(output, {input: processed_image})[0][1]
        except Exception as e:
            print(f"ClassifierError {e}")
            raise Exception(f'classifier error! {e}')
      
        r = {"probability": round(float(acc_percentage),4)}
        data["predictions"].append(r)

        #print("predicted class: {}, acc: {}".format(pred_class, acc_percentage))
        print(data)
        
        return data


