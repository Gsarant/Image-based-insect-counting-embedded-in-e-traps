
import os
from tqdm import tqdm
import torch
from torchsummary import summary
from imutils import paths
import time
import shutil
import cv2
import math
import copy
import sys
sys.path.append('..')
from gsutils import init_logger, a, load_images_inference, saved_image_path, mape_fun
sys.path.append('../train')
import count_regression_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_params = [
    # Overlap ResNet18
    {'test_image_path': '../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
     'model_name': 'Count_Regression_Helicoverpa_resnet18_10k_overlap',
     'size': (224, 224),
     'round': True,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.74656951-valloss 0.7465695121458599- CountRegr_Helicoverpa_resnet18_HVGA_ep_104.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_resnet18_10k_overlap'},
    # Overlap ResNet50
    {'test_image_path': '../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
     'model_name': 'Count_Regression_Helicoverpa_resnet50_10k_overlap',
     'size': (224, 224),
     'round': True,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.69831866-valloss 0.6983186623879841- CountRegr_Helicoverpa_resnet50_HVGA_ep_113.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_resnet50_10k_overlap'},
    # Overlap vgg16
    {'test_image_path': '../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
     'model_name': 'Count_Regression_Helicoverpa_vgg16_10k_overlap',
     'size': (224, 224),
     'round': True,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k_2/Helicoverpa_armigera_vgg16_HVGA-1.11196765-valloss 1.1119676530361176- CountRegr_Helicoverpa_vgg16_HVGA_ep_63.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_vgg16_10k_overlap'}, 
    # Overlap Custom Model
    {'test_image_path': '../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
     'model_name': 'Count_Regression_Helicoverpa_large_10k_overlap',
     'size': (480, 320),
     'round': True,
     'create_image': True,
     'grey': True,
     'enable': False,
     'load_saved_parameters': '../train/ckpt_10k_new/Plodia_interpunctella_resnet_HVGA-1.96481650-valloss 1.9648165039394212- CountRegr_Plodia_interpunctella_resnet_HVGA_ep_102.pt',
     'test_save_images_path': 'tests/Count_Regression_10k_new/Helicoverpa_armigera_large_10k_overlap'},
    # Helicoverpa ResNet18
    {'test_image_path': '../../data/Helicoverpa_armigera',
     'model_name': 'Count_Regression_Helicoverpa_resnet18_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.74656951-valloss 0.7465695121458599- CountRegr_Helicoverpa_resnet18_HVGA_ep_104.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_resnet18_10k'},
    # Helicoverpa ResNet50
    {'test_image_path': '../../data/Helicoverpa_armigera',
     'model_name': 'Count_Regression_Helicoverpa_resnet50_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Helicoverpa_armigera_resnet_HVGA-0.69831866-valloss 0.6983186623879841- CountRegr_Helicoverpa_resnet50_HVGA_ep_113.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_resnet50_10k'},
    # Helicoverpa vgg16
    {'test_image_path': '../../data/Helicoverpa_armigera',
     'model_name': 'Count_Regression_Helicoverpa_vgg16_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k_2/Helicoverpa_armigera_vgg16_HVGA-1.11196765-valloss 1.1119676530361176- CountRegr_Helicoverpa_vgg16_HVGA_ep_63.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_vgg16_10k'},
    # Helicoverpa Custom Model
    {'test_image_path': '../../data/Helicoverpa_armigera',
     'model_name': 'Count_Regression_Helicoverpa_large_10k',
     'size': (480, 320),
     'round': False,
     'create_image': True,
     'grey': True,
     'enable': False,
     'load_saved_parameters': '../train/ckpt_10k/Helicoverpa_armigera_grey_large_HVGA-18.28486204-valloss 18.284862043799425- CountRegr_Helicoverpa_grey_large_HVGA_ep_93.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Helicoverpa_armigera_large_10k'},
    # Plodia ResNeτ18
    {'test_image_path': '../../data/Plodia_interpunctella',
     'model_name': 'Count_Regression_Plodia_interpunctella_resnet18_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Plodia_interpunctella_resnet_HVGA-2.07001807-valloss 2.0700180748234622- CountRegr_Plodia_interpunctella_resnet18_HVGA_ep_158.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Plodia_interpunctella_resnet18_10k'},
    # Plodia ResNet50
    {'test_image_path': '../../data/Plodia_interpunctella',
     'model_name': 'Count_Regression_Plodia_interpunctella_resnet50_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k/Plodia_interpunctella_resnet_HVGA-3.17570898-valloss 3.1757089780724566- CountRegr_Plodia_interpunctella_resnet50_HVGA_ep_49.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Plodia_interpunctella_resnet50_10k'},
    # Plodia vgg16
    {'test_image_path': '../../data/Plodia_interpunctella',
     'model_name': 'Count_Regression_Plodia_interpunctella_vgg16_10k',
     'size': (224, 224),
     'round': False,
     'create_image': True,
     'grey': False,
     'enable': True,
     'load_saved_parameters': '../train/ckpt_10k_2/Plodia_interpunctella_vgg16_HVGA-2.08870671-valloss 2.088706708991009- CountRegr_Plodia_interpunctella_vgg16_HVGA_ep_158.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Plodia_interpunctella_vgg16_10k'},
    # Plodia Custom Model
    {'test_image_path': '../../data/Plodia_interpunctella',
     'model_name': 'Count_Regression_Plodia_interpunctella_large_10k',
     'size': (480, 320),
     'round': False,
     'create_image': True,
     'grey': True,
     'enable': False,
     'load_saved_parameters': '../train/ckpt_10k/Plodia_interpunctella_grey_large_HVGA-48.27876088-valloss 48.27876087535511- CountRegr_Plodia_interpunctella_lage_HVGA_ep_2.pt',
     'test_save_images_path': 'tests/Count_Regression_10k/Plodia_interpunctella_large_10k'},

    # Plodia Small Custom Model
    {'test_image_path': '../../data/Plodia_interpunctella',
     'model_name': 'Count_Regression_Plodia_interpunctella_grey_HVGA_small_last_20k',
     'size': (480, 320),
     'round': False,
     'create_image': False,
     'grey': True,
     'enable': False,
     'load_saved_parameters': '../train/ckpt_20k/Plodia_interpunctella_grey_small_HVGA-3.1565-last-Plodia_interpunctella_grey_small_HVGA_ep_50.pt',
     'test_save_images_path': 'tests/Count_Regression_20k/Plodia_interpunctella_grey_HVGA_small_last_20k'},
]
LOG = 'tests/logs_10k'
try:
    shutil.rmtree(LOG)
except:
    pass
os.makedirs(LOG)
# general results
gen_res_logger = init_logger(LOG, 'count_regression_genResults')
for test_param in test_params:
    if test_param['enable'] == True:

        # Load Logger
        test_logger = init_logger(LOG, test_param['model_name'])

        try:
            shutil.rmtree(test_param['test_save_images_path'])
        except:
            pass
        os.makedirs(test_param['test_save_images_path'])

        # Load model
        model = torch.load(test_param['load_saved_parameters'])
      #  print(model)
      #  summary(model,(3,224,224))
        model.to(device)
        model.eval()

        # Init Images
        full_image_path = list(paths.list_images(
            test_param['test_image_path']))
        # init bar
        test_pbar = tqdm(range(len(full_image_path)),
                         f"Test in progress : {test_param['test_save_images_path']} Model :{test_param['model_name']} Checkpoint : {test_param['load_saved_parameters']}")
        # Head Log
        test_logger.info(
            f"Test in progress : {test_param['test_save_images_path']} Model :{test_param['model_name']} Checkpoint : {test_param['load_saved_parameters']}")
        test_logger.info(f"image , Prediction , Label , Predict time")

        # Init metrics
        percent_acceptable_error = 0.2
        acceptable_errors_1_20 = 0
        sum_mae_1_20 = 0.0
        sum_a_1_20 = 0
        sum_mape_1_20 = 0
        sum_mse_1_20 = 0
        sum_time_1_20 = 0
        count_1_20 = 0

        acceptable_errors_50_100 = 0
        sum_mae_50_100 = 0.0
        sum_a_50_100 = 0
        sum_mape_50_100 = 0
        sum_mse_50_100 = 0
        sum_time_50_100 = 0
        count_50_100 = 0
        for img in full_image_path:
            # Init timer
            start_im_time = time.time()
            # Load image
            image, image1 = load_images_inference(
                img, test_param['size'], test_param['grey'])
            #image,image1=load_images_inference(img,test_param['size'],True,True  )
            # Read Label from image filename
            label = int(img.split(".")[-2].split("_")[-1:][0])
            # Prepair image and inference from model
            image = torch.tensor(image)
            image = image.to(device)
            output = model(image)

            # Proccessing output and predict label
            pred_label = output.data.cpu().detach().float().item()
            if test_param['round'] == True:
                pred_label = round(pred_label)

            # Update metrics
            if label > 20:
                sum_a_50_100 += a(pred_label, label)
                sum_mae_50_100 += abs(label-pred_label)
                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_50_100 += 1
                sum_mse_50_100 += math.pow((label-pred_label), 2)
                sum_time_50_100 += (time.time()-start_im_time)*1000
                sum_mape_50_100 += mape_fun(pred_label, label)
                count_50_100 += 1
            else:
                sum_a_1_20 += a(pred_label, label)
                sum_mae_1_20 += abs(label-pred_label)

                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_1_20 += 1

                sum_mse_1_20 += math.pow((label-pred_label), 2)
                sum_time_1_20 += (time.time()-start_im_time)*1000
                sum_mape_1_20 += mape_fun(pred_label, label)
                count_1_20 += 1

            # Update bar
            test_pbar.update()
            # Update log
            #test_logger.info(f"{img},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")
            test_logger.info(
                f"{img}, {pred_label:.2f} , {label:.0f} , {(time.time()-start_im_time)*1000:.1f}")

            if test_param['create_image'] and image1 is not None:
                # summary(model,(3,224,224))
                # c=list(list(model.children())[0].layer4)[2].conv1.weight
                # c=list(model.children())[0].avgpool
                # print(c)
                # print(model)

                image1 = cv2.putText(
                    image1, f"Regrs: {pred_label:.1f}/{label}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imwrite(saved_image_path(
                    test_param['test_save_images_path'], img, 'dest'), image1)
       # Final metrics
        count = count_1_20 + count_50_100
        mae = (sum_mae_1_20 + sum_mae_50_100)/count
        avg_a = (sum_a_1_20 + sum_a_50_100)/count
        mape = (sum_mape_1_20+sum_mape_50_100)/count
        mse = (sum_mse_1_20+sum_mse_50_100)/count
        rmse = math.sqrt(mse)
        sum_time = sum_time_1_20 + sum_time_50_100
        avg_acceptable_errors = (
            acceptable_errors_1_20+acceptable_errors_50_100)/count
        test_logger.info(
            f"{test_param['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
        gen_res_logger.info(
            f" {test_param['model_name']} \t \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.3f}% \t RMSE:{rmse:.4f} \t AE {avg_acceptable_errors:.2f}% \t AVG_TIME {(sum_time/(count)):.1f}")
        if count_1_20 > 0:
            gen_res_logger.info(f" {test_param['model_name']} 1-20 \t A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.3f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.4f} \t AE {acceptable_errors_1_20/count_1_20:.2f}% \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
        if count_50_100 > 0:
            gen_res_logger.info(f" {test_param['model_name']} 50-100 \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.3f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.4f} \t AE {acceptable_errors_50_100/count_50_100:.2f}% \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")
