import os
from convert_pt_onnx import Convert_pt_onnx
from convert_onnx_tf_tflite import Convert_onnx_tf_tflite
import shutil

export_param=[
    
             {
              'val_image_path':'../../data/datasets_Helicoverpa_armigera_10k/images/val',
              'model_name':'Yolov7_Helicoverpa_HVGA_10k',
              'size':(480,620),
              'grey':False,
              'load_onnx_file':'../Yolov7runs_10k/Helicoverpa-armigera/weights/best.onnx',
              #'load_saved_parameters':'/home/giannis/paper2/yolotrain/Yolov7runs/Helicoverpa-armigera/weights/best.pt',
              'export_onnx_path':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_onnx',
              'export_tf_path':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_tf'},
            {
               'val_image_path':'../../data/datasets_Plodia_interpunctella_10k/images/val',
               'model_name':'Yolov7_plodia_HVGA_10k',
               'size':(480,320),
               'grey':False,
               'load_onnx_file':'../Yolov7runs_10k/Plodia/weights/best.onnx',
               #'load_saved_parameters':'/home/giannis/paper2/MyCrowdCounting/insect_CSRNet/src/ckpt/Plodia_interpunctella_grey-2.5722-CSRNET_plodia_grey_ep_192.pt',
               'export_onnx_path':'../export/yolov7_10k/Plodia_interpunctella_HVGA_onnx',
               'export_tf_path':'../export/yolov7_10k/Plodia_interpunctella_HVGA__tf'},
            
            ]
for export_param in export_param:
    
    
    
    #remove_files(export_param['export_onnx_path'])
    try:
        shutil.rmtree(export_param['export_tf_path'])
    except:
        pass    
    #create_folders(export_param['export_onnx_path'])
    os.makedirs(export_param['export_tf_path'])
    #convert_pt_onnx=Convert_pt_onnx(export_param['load_saved_parameters'],
    #                  export_param['export_onnx_path'],
    #                  export_param['size'],
    #                  export_param['grey'])
    #onnx_file_path=convert_pt_onnx.convert()
    onnx_file_path=export_param['load_onnx_file']
    
    convert_onnx_tf_tflite=Convert_onnx_tf_tflite(onnx_file_path,
                                        export_param['export_tf_path'],
                                        export_param['val_image_path'],
                                        export_param['size'],
                                        export_param['grey']  )
    
    convert_onnx_tf_tflite.convert_onnx_tf()
    convert_onnx_tf_tflite.convert_tf_tflite(False)
