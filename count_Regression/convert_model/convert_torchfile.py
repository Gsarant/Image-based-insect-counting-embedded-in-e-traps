import os
from convert_pt_onnx import Convert_pt_onnx
from convert_onnx_tf_tflite import Convert_onnx_tf_tflite
import shutil

export_param=[
            # Helicoverpa ResNet18
            {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera/images/val',
              'model_name':'CSRNet_Helicoverpa_resnet18',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':True,
              'load_saved_parameters':'../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.74656951-valloss 0.7465695121458599- CountRegr_Helicoverpa_resnet18_HVGA_ep_104.pt',
              'export_onnx_path':'export_10k/Count_Regression/Helicoverpa_armigera_resnet18_onnx',
              'export_tf_path':'export_10k/Count_Regression/Helicoverpa_armigera_resnet18_tf'},
             # Helicoverpa ResNet50
            {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera/images/val',
              'model_name':'CSRNet_Helicoverpa_resnet50',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':True,
              'load_saved_parameters':'../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.69831866-valloss 0.6983186623879841- CountRegr_Helicoverpa_resnet50_HVGA_ep_113.pt',
              'export_onnx_path':'export_10k/Count_Regression/Helicoverpa_armigera_resnet50_onnx',
              'export_tf_path':'export_10k/Count_Regression/Helicoverpa_armigera_resnet50_tf'},
            # Helicoverpa vgg16
             {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera/images/val',
              'model_name':'CSRNet_Helicoverpa_vgg18',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':False,
              'load_saved_parameters':'../train/ckpt_10k_2/Helicoverpa_armigera_vgg16_HVGA-1.11196765-valloss 1.1119676530361176- CountRegr_Helicoverpa_vgg16_HVGA_ep_63.pt',
              'export_onnx_path':'export_10k/Count_Regression/Helicoverpa_armigera_vgg16_onnx',
              'export_tf_path':'export_10k/Count_Regression/Helicoverpa_armigera_vgg16_tf'},

            # Plodia ResNet18
            {'test_image_path':'../../data/Plodia_interpunctella',
              'val_image_path':'../../data/datasets_Plodia_interpunctella/images/val',
              'model_name':'CSRNet_plodia_resnet18',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':True,
              'load_saved_parameters':'../train/ckpt_10k/Plodia_interpunctella_resnet_HVGA-2.07001807-valloss 2.0700180748234622- CountRegr_Plodia_interpunctella_resnet18_HVGA_ep_158.pt',
              'export_onnx_path':'export_10k/Count_Regression/Plodia_interpunctella_resnet18_onnx',
              'export_tf_path':'export_10k/Count_Regression/Plodia_interpunctella_resnet18_tf'},
            # Plodia ResNet50
            {'test_image_path':'../../data/Plodia_interpunctella',
              'val_image_path':'../../data/datasets_Plodia_interpunctella/images/val',
              'model_name':'CSRNet_plodia_resnet50',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':True,
              'load_saved_parameters':'../train/ckpt_10k/Plodia_interpunctella_resnet_HVGA-3.17570898-valloss 3.1757089780724566- CountRegr_Plodia_interpunctella_resnet50_HVGA_ep_49.pt',
              'export_onnx_path':'export_10k/Count_Regression/Plodia_interpunctella_resnet50_onnx',
              'export_tf_path':'export_10k/Count_Regression/Plodia_interpunctella_resnet50_tf'},
             # Plodia vgg16
            {'test_image_path':'../../data/Plodia_interpunctella',
              'val_image_path':'../../data/datasets_Plodia_interpunctella/images/val',
              'model_name':'CSRNet_plodia_vgg16',
              'size':(224,224),
              'grey':False,
              'quantization':False,
              'enable':False,
              'load_saved_parameters':'../train/ckpt_10k_2/Plodia_interpunctella_vgg16_HVGA-2.08870671-valloss 2.088706708991009- CountRegr_Plodia_interpunctella_vgg16_HVGA_ep_158.pt',
              'export_onnx_path':'export_10k/Count_Regression/Plodia_interpunctella_vgg16_onnx',
              'export_tf_path':'export_10k/Count_Regression/Plodia_interpunctella_vgg16_tf'},
           
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
