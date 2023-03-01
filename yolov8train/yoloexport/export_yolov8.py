import os
import sys
sys.path.append('../../ultralytics')
from ultralytics import YOLO
import torch.onnx
import onnx
import shutil
from convert_onnx_tf_tflite import Convert_onnx_tf_tflite

export_params=[
    
             {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera_10k/images/val',
              'model_name':'Yolov8_Helicoverpa_HVGA_10k',
              'size':(480,620),
              'grey':False,
              'load_saved_parameters':'/home/giannis/paper2/yolov8train/runs/detect/Yolov8runs_10k/Helicoverpa-armigera2/weights/best.pt',
              'export_tf_path':'export/yolov8_10k/Helicoverpa_armigera_HVGA_tf'},
            {
               'val_image_path':'../../data/datasets_Plodia_interpunctella_10k/images/val',
               'model_name':'Yolov8_plodia_HVGA_10k',
               'size':(480,320),
               'grey':False,
               'load_saved_parameters':'/home/giannis/paper2/yolov8train/runs/detect/Yolov8runs_10k/Plodia/weights/best.pt',
               'export_tf_path':'export/yolov8_10k/Plodia_interpunctella_HVGA__tf'},
            
            ]

def main():
    for export_param in export_params:
        try:
            shutil.rmtree(export_param['export_tf_path'])
        except:
            pass    
        os.makedirs(export_param['export_tf_path'])
        model = YOLO(export_param["load_saved_parameters"])
        model.fuse()
        #success = model.export(format="tflite")  
        success = model.export(format="tflite")  
        

      #  image_name=export_param["load_saved_parameters"].split(os.sep)[-1]
      #  a=export_param["load_saved_parameters"].split(os.sep)[:-1]
      #  tflite_name=f'{image_name.split(".")[-2]}.onnx'
      #  tflite_name=os.path.join('/',*a,tflite_name)
      #  convert_onnx_tf_tflite=Convert_onnx_tf_tflite(
      #                                           tflite_name,
      #                                           export_param["export_tf_path"],
      #                                           export_param["val_image_path"],
      #                                           export_param["size"],
      #                                           export_param["grey"])
      #  convert_onnx_tf_tflite.convert_onnx_tf()
      #  convert_onnx_tf_tflite.convert_tf_tflite(False)

if __name__ == "__main__":
    main()
