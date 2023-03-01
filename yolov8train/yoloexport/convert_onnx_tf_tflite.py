import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from imutils import paths
from datetime import datetime
import cv2
import os

class Convert_onnx_tf_tflite(object):
    def __init__(self,onnx_file_path,
                export_tf_path,
                val_images_path=None,
                size=(800,800),
                grey=False):
        self.onnx_file_path=onnx_file_path
        self.val_images_path=val_images_path
        self.export_tf_path=export_tf_path
        self.full_image_path=list(paths.list_images(val_images_path))
        self.size=size
        self.grey=grey
    
    def create_new_converted_file(self,save_parh,orig_file_path,dest_ext='tflite'):
        file_name=orig_file_path.split(os.sep)[-1:][0]
        file_name_without_ext=file_name.split('.')[0]
        image_name=f'{file_name_without_ext}.{dest_ext}'
        return os.path.join(save_parh,image_name)

    def load_images(self,imagePath):
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        if self.grey:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, self.size) / 255.0
        image = tf.transpose(image,[0,1,2])
        return image
    
    def create_dataset(self):

        AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE = 32
        self.train_ds = tf.data.Dataset.from_tensor_slices( self.full_image_path).map(self.load_images, num_parallel_calls=AUTOTUNE)
        self.train_ds = self.train_ds.cache()
        self.train_ds = self.train_ds.batch(BATCH_SIZE)
        self.train_ds = self.train_ds.prefetch(AUTOTUNE)
        self.load_images(self.full_image_path[0])

  

    def representative_data_gen2(self):
        for setimage   in self.train_ds.take(50):
            for index in range(len(setimage)):
                image = setimage[index]
                #image = (np.float32(image) / 255.0)
                image=np.expand_dims(image, axis=0) 
                yield [image.astype(np.float32)]
    
    def representative_data_gen(self):
        for image_name in self.full_image_path[:50]:
            image=cv2.imread(image_name)

            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

            if self.grey:
                image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
                image=np.expand_dims(image, axis=0)   
            else:
                image=np.transpose(image,[2,0,1])     
            image = (np.float32(image) / 255.0)
            
            image=np.expand_dims(image, axis=0) 
            yield [image.astype(np.float32)]

    def convert_onnx_tf(self):
        onnx_model = onnx.load(self.onnx_file_path)
        onnx.checker.check_model(onnx_model)
        tf_rep = prepare(onnx_model)
        tf_model_path=self.export_tf_path
        tf_rep.export_graph(tf_model_path)
        return tf_model_path
    
    def convert_tf_tflite(self,quantization=True):

        converter = tf.lite.TFLiteConverter.from_saved_model(self.export_tf_path)
        #tflite_model = converter.convert()
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization:
            if not self.representative_data_gen is None:
                # self.create_dataset()
                converter.representative_dataset =self.representative_data_gen

        
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
            converter.target_spec.supported_types = [tf.int8]
        
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        tflite_model_quant = converter.convert()

        path_saved_model_tflite_quant=self.create_new_converted_file(self.export_tf_path,self.onnx_file_path,'tflite')
        with open(path_saved_model_tflite_quant, 'wb') as f:
            f.write(tflite_model_quant)
