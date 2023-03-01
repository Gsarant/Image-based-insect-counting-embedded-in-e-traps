import torch.onnx
import onnx
import os

class Convert_pt_onnx(object):
    def __init__(self,load_saved_parameters,export_onnx_path,size,grey):
        self.load_saved_parameters=load_saved_parameters
        self.export_onnx_path=export_onnx_path
        self.size=size
        self.grey=grey
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_new_converted_file(self,save_parh,orig_file_path,dest_ext='onnx'):
        file_name=orig_file_path.split(os.sep)[-1:][0]
        file_name_without_ext=file_name.split('.')[0]
        image_name=f'{file_name_without_ext}.{dest_ext}'
        return os.path.join(save_parh,image_name)
    
    
    
    def convert(self):
        model = torch.load(self.load_saved_parameters)
        #checkpoint = torch.load(export_param['load_saved_parameters'])
        #model = export_param['model']
        #a=checkpoint.state_dict()
        #model.load_state_dict(a)
    
        model.to(self.device)
        model.eval()
        generated_input =torch.randn(1, 1 if self.grey else 3, self.size[0], self.size[1]).to(self.device)
        onnx_file_path=self.create_new_converted_file(self.export_onnx_path,self.load_saved_parameters,'onnx')
        torch.onnx.export(model,
                      generated_input,   
                      onnx_file_path,
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                     )
        onnx_model = onnx.load(onnx_file_path)
        onnx.checker.check_model(onnx_model)
        return onnx_file_path
        print(onnx.helper.printable_graph(onnx_model.graph))