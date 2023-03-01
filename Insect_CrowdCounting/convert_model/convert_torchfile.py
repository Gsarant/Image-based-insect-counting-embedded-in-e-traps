import os
from convert_pt_onnx import Convert_pt_onnx
from convert_onnx_tf_tflite import Convert_onnx_tf_tflite
import shutil
import sys
sys.path.append('../train_insect_CSRNet')
from csrnet import CSRNet,CSRNet_small,CSRNet_medium,CSRNet_medium_color

export_param=[
             {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera_10k/images/val',
              'model_name':'CSRNet_Helicoverpa_color_HVGA',
              'size':(480,320),
              'grey':False,
              'quantization':False,
              'enable':False,
              'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Helicoverpa_armigera_color_HVGA_10k-valloss 0.00045534-CSRNET_Helicoverpa_color_HVGA_10k_ep_297.pt',
              'export_onnx_path':'export_10k/CSRNet/Helicoverpa_armigera_color_HVGA_onnx',
              'export_tf_path':'export_10k/CSRNet/Helicoverpa_armigera_color_HVGA_tf'},
            {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera_10k/images/val',
              'model_name':'CSRNet_Helicoverpa_medium_color_HVGA',
              'size':(480,320),
              'grey':False,
              'quantization':False,
              'enable':False,
              'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Helicoverpa_armigera_color_medium_HVGA_10k-0.00072860-last-Helicoverpa_armigera_color_medium_HVGA_10k_ep_60.pt',
              'export_onnx_path':'export_10k/CSRNet/Helicoverpa_armigera_medium_color_HVGA_onnx',
              'export_tf_path':'export_10k/CSRNet/Helicoverpa_armigera_medium_color_HVGA_tf'},
            
           {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera_10k/images/val',
              'model_name':'CSRNet_Helicoverpa_color_HVGA_quant',
              'size':(480,320),
              'grey':False,
              'quantization':True,
              'enable':True,
              'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Helicoverpa_armigera_color_HVGA_10k-valloss 0.00045534-CSRNET_Helicoverpa_color_HVGA_10k_ep_297.pt',
              'export_onnx_path':'export_10k_quant/CSRNet/Helicoverpa_armigera_color_HVGA_onnx_quant',
              'export_tf_path':'export_10k_quant/CSRNet/Helicoverpa_armigera_color_HVGA_tf_quant'},





           {  'val_image_path':'../../data/datasets_Plodia_interpunctella_10k/images/val',
              'model_name':'CSRNet_plodia_color_HVGA',
              'size':(480,320),
              'grey':False,
              'quantization':False,
              'enable':False,
              'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Plodia_interpunctella_color_HVGA_10k-valloss 0.00082795-CSRNET_Plodia_color_HVGA_10k_ep_296.pt',
              'export_onnx_path':'export_10k/CSRNet/Plodia_interpunctella_color_HVGA_onnx',
              'export_tf_path':'export_10k/CSRNet/Plodia_interpunctella_color_HVGA_tf'},
            
            { 'val_image_path':'../../data/datasets_Plodia_interpunctella_10k/images/val',
              'model_name':'CSRNet_plodia_medium_color_HVGA',
              'size':(480,320),
              'grey':False,
              'quantization':False,
              'enable':False,
               'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Plodia_interpunctella_color_medium_HVGA_10k-0.00088079-last-Plodia_interpunctella_color_medium_HVGA_10k_ep_84.pt',
              'export_onnx_path':'export_10k/CSRNet/Plodia_interpunctella_medium_color_HVGA_onnx',
              'export_tf_path':'export_10k/CSRNet/Plodia_interpunctella_medium_color_HVGA_tf'},
              
            { 'val_image_path':'../../data/datasets_Plodia_interpunctella_10k/images/val',
              'model_name':'CSRNet_plodia_color_quant_HVGA',
              'size':(480,320),
              'grey':False,
              'quantization':True,
              'enable':True,
              'load_saved_parameters':'../train_insect_CSRNet/ckpt_10k_5/Plodia_interpunctella_color_HVGA_10k-valloss 0.00082795-CSRNET_Plodia_color_HVGA_10k_ep_296.pt',
              'export_onnx_path':'export_10k_quant/CSRNet/Plodia_interpunctella_color_HVGA_onnx',
              'export_tf_path':'export_10k_quant/CSRNet/Plodia_interpunctella_color_HVGA_tf'},
            
            ]
for export_param in export_param:
  if  export_param['enable']==True:
    try:
      shutil.rmtree(export_param['export_onnx_path'])
    except:
      pass    
    os.makedirs(export_param['export_onnx_path'])
    try:
      shutil.rmtree(export_param['export_onnx_path'])
    except:
      pass    
    os.makedirs(export_param['export_onnx_path'])
    
    convert_pt_onnx=Convert_pt_onnx(export_param['load_saved_parameters'],
                      export_param['export_onnx_path'],
                      export_param['size'],
                      export_param['grey'])
    onnx_file_path=convert_pt_onnx.convert()
  
  
    convert_onnx_tf_tflite=Convert_onnx_tf_tflite(onnx_file_path,
                                        export_param['export_tf_path'],
                                        export_param['val_image_path'],
                                        export_param['size'],
                                        export_param['grey']  )
    
    convert_onnx_tf_tflite.convert_onnx_tf()
    convert_onnx_tf_tflite.convert_tf_tflite(export_param['quantization'])
