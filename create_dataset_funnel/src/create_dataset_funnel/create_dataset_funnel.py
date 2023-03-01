import numpy as np
import cv2
import os
import random
from create_dataset_funnel.image_insects_funnel import Image_Insects_Funnel
from imutils import paths
from datetime import datetime
import time
import shutil
from multiprocessing import Pool,cpu_count
from sympy.geometry import Point, Circle, intersection, Polygon

# Create Custom Dataset for Train and Validation


class CreateDatasetFunnel:
    def __init__(self, path_funnels, path_dirtes, path_insects, path_dataset, foldertype='train', remove_old_dataset=False, groups_insect_per_image=[1], images_per_group=2, dest_image_width=1664, dest_image_height=1232):
        random.seed = 10
        self.show_images = True
        self.path_funnels = path_funnels
        self.path_dirtes = path_dirtes
        self.path_insects = path_insects
        self.dest_image_width = dest_image_width
        self.dest_image_height = dest_image_height
        self.path_dataset = path_dataset
        self.images_path = os.path.join(path_dataset, 'images', foldertype)
        self.annotations_path = os.path.join(path_dataset, 'labels', foldertype)
        if remove_old_dataset:
            self.__remove_files__(self.images_path)
            self.__remove_files__(self.annotations_path)
        self.__create_folders__(self.path_dataset)
        self.__create_folders__(self.images_path)
        self.__create_folders__(self.annotations_path)
        self.groups_insect_per_image = groups_insect_per_image
        self.images_per_group = images_per_group

    # Create folders from  given paths
    def __create_folders__(self, folders):
        folders_array = folders.split(os.sep)
        for f in range(len(folders_array)):
            cur_folder = os.path.sep.join(folders_array[:(f+1)])
            try:
                os.makedirs(cur_folder)
            except:
                pass
        

    # Remove files from a folder
    def __remove_files__(self, folder):
        try:
            shutil.rmtree(folder)
        except:
            pass
   
    # Class Destroy
    def __del__(self):
        try:
            cv2.waitKey(20000)
            cv2.destroyAllWindows()
        except:
            pass

    # Show an image in a window
    def __show_image(self, text, image):
        cv2.imshow(text, image)
        cv2.waitKey(100)

    # Save a custom image to specific folder
    def __save_insect_with_funnel(self, image, full_insect_funnel_path, full_insect_path, resize=True):
        self.__create_folders__(full_insect_funnel_path)
        list_full_insect_path = full_insect_path.split('/')
        full_insect_funnel_iamge_path = os.path.join(
            full_insect_funnel_path, list_full_insect_path[-1])
        self.__save_image(image, full_insect_funnel_iamge_path, False)
        

    # Resize and save an image
    def __save_image(self, image, path_image, resize=True):
        try:
            if resize:
                image = cv2.resize(
                    image, (self.dest_image_width, self.dest_image_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_image, image)
        except:
            print(f'{path_image} not saved')

    # Normalize the position of an object
    def __norm(self, x, y, w, h, image_w, image_h):
        x = float(x)+int(float(w)/2)
        x = float(x)/image_w
        y = float(y)+int(float(h)/2)
        y = float(y)/image_h
        w = float(w)/image_w
        h = float(h)/image_h
        return x, y, w, h

    # Create a dataset
    def create_dataset(self, overlap=False):
        # Groups with numbers of insect per image
        for index_group in self.groups_insect_per_image:
            # Number of images per group
            for index_insects in range(self.images_per_group):
                image_insects_funnel = Image_Insects_Funnel(
                    self.path_funnels, self.path_insects)
                image, count_overlay, points_array = image_insects_funnel.create_image(
                    index_group, overlap)
                self.__save_insect_with_funnel(
                    image, self.path_dataset, f'{index_insects+1}_{index_group}.jpg')

    def __check_clear_area(self,list_rect,rect):
        x1=rect[0]
        x2=rect[0]+rect[2]
        y1=rect[1]
        y2=rect[1]+rect[3]
        for tmp_rect in list_rect:
            t_x1 =tmp_rect[0]
            t_x2=tmp_rect[0]+tmp_rect[2]
            t_y1=tmp_rect[1]
            t_y2=tmp_rect[1]+tmp_rect[3]
            #if ((x1 >= t_x1 and x1 <= t_x2) and (y1 >= t_y1 and y1 <= t_y2))  \
            #   or \
            #   ((x2 >= t_x1 and x2 <= t_x2) and (y1 >= t_y1 and y1 <= t_y2)) \
            #   or \
            #   ((x1 >= t_x1 and x1 <= t_x2) and (y2 >= t_y1 and y2 <= t_y2)) \
            #   or \
            #   ((x2 >= t_x1 and x2 <= t_x2) and (y2 >= t_y1 and y2 <= t_y2)) : 
            if (x1>=t_x2 or x2<=t_x1) ==False and (y2<=t_y1 or y1>=t_y2)==False:
                return False
        return True

    def __check_clear_area3(self,list_rect,rect):   
        x1=rect[0]
        x2=rect[0]+rect[2]
        y1=rect[1]
        y2=rect[1]+rect[3] 
        r=Polygon(Point(x1, y1),Point(x2, y1),
                Point(x2,y2), Point(x1,y2))    
        for tmp_rect in list_rect:
            t_x1 =tmp_rect[0]
            t_x2=tmp_rect[0]+tmp_rect[2]
            t_y1=tmp_rect[1]
            t_y2=tmp_rect[1]+tmp_rect[3]
            t_r=Polygon(Point(t_x1, t_y1),Point(t_x2, t_y1),
                Point(t_x2,t_y2), Point(t_x1,t_y2)) 
            if  len(t_r.intersection(r))>0:
                return False
        return True        


    def __create_empty_box(self,debris_bound_boxes,bound_box,num):
        for index in range(num):
            count=0
            while True:
                count +=1
                w=random.randint(1,10000)/random.randint(50000,100000)
                h=random.randint(1,10000)/random.randint(50000,100000)
                x=random.randint(1,1000)/random.randint(1000,5000)
                y=random.randint(1,1000)/random.randint(1000,5000)
                rect=[x,y,w,h]
                if  self.__check_clear_area(bound_box,rect)==True:
                    debris_bound_boxes.append(rect)
                    break
                if count > 5:
                    break
                





    def __create_image(self, Count_images, image_overlap, index_images, index_group, overlap=False,process_id=0):
        start_time = time.time()
        # Funnel image
        image_insects_funnel = Image_Insects_Funnel(
            self.path_funnels, self.path_dirtes, self.path_insects, )
        # Funnel Image with dirtes
        image, count_overlap, points_array, dirtes_points_array = image_insects_funnel.create_image(
            index_group, overlap)
        if count_overlap > 0:
            image_overlap += 1

        bound_boxes = []
        for points in points_array:
            bound_boxes.append(self.__norm(
                points[0], points[1], points[2]-points[0], points[3]-points[1], image.shape[1], image.shape[0]))

        debris_bound_boxes = []
        for points in dirtes_points_array:
            debris_bound_boxes.append(self.__norm(
                points[0], points[1], points[2]-points[0], points[3]-points[1], image.shape[1], image.shape[0]))
        image=cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
        # Save custom image
        self.__save_insect_with_funnel(
            image, self.images_path, f'f_{process_id}_{index_images+1}_{index_group}.jpg')
        Count_images += 1
       
        print(f' image f_{process_id}_{index_images+1}_{index_group}.jpg  time {(time.time()-start_time)//60} min {(time.time()-start_time)%60:.2f} sec {datetime.now().strftime("%H:%M:%S")}')
        
        self.__create_empty_box(debris_bound_boxes,bound_boxes,abs(index_group-len(debris_bound_boxes))//2)
        
        # Save normalization position in txt file for Yolo
        with open(os.path.join(self.annotations_path, f'f_{process_id}_{index_images+1}_{index_group}.txt'), 'w') as f:
            for bound_box in bound_boxes:
                f.write('1  ')
                for s in bound_box:
                    f.write(f'{s:.6f}  ')
                f.write('\n')
            for bound_box in debris_bound_boxes:
                f.write('0  ')
                for s in bound_box:
                    f.write(f'{s:.6f}  ')
                f.write('\n')

        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")

        with open(os.path.join(self.path_dataset, f'insects_{len(self.groups_insect_per_image)}_test_new.txt'), 'a') as f:
            f.write(
                f'{current_time} {self.images_path}/f_{process_id}_{index_images+1}_{index_group}.jpg, {count_overlap}/{index_group}\n')
        return Count_images, image_overlap

    # Create Dataset with labelling for Yolo
    def create_dataset_yolov5(self, overlap=False):
        Count_images = 0
        image_overlap = 0
        # Groups with numbers of insect per image

        for index_group in self.groups_insect_per_image:
            # Number of images per group
            for index_images in range(self.images_per_group):
                Count_images, image_overlap=self.__create_image(Count_images, image_overlap,
                                    index_images, index_group, overlap)
        with open(os.path.join(self.path_dataset, f'insects_{len(self.groups_insect_per_image)}_test_new.txt'), 'a') as f:
            f.write(
                f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')
        print(
            f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')

    # Create Dataset with labelling for Yolo Random number insect per image
    def create_dataset_yolo_norm_rand(self, low, high, size, overlap=False):
        Count_images = 0
        image_overlap = 0
        list_insect_in_image = np.random.randint(low=low, high=high, size=size)
        process_data=[]
        process_num=cpu_count()
        multiplier=int(len(list_insect_in_image)//process_num)
        for i in range(process_num-1):
            process_data.append([i,list_insect_in_image[i*multiplier:(i+1)*multiplier],overlap])
        if int(len(list_insect_in_image)%process_num)>0:
            process_data.append([i+1,list_insect_in_image[(i+1)*multiplier:],overlap])
        p=Pool(process_num)
        return_val=p.map(self.create_dataset_yolo_norm_rand_process,process_data)
        Count_images=0
        image_overlap=0
        for r_v in return_val:
            Count_images +=r_v[0]
            image_overlap +=r_v[1]
        if Count_images>0:
            with open(os.path.join(self.path_dataset, f'insects_{len(self.groups_insect_per_image)}_test_new.txt'), 'a') as f:
                f.write(f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')
            print(f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')
        
    def create_dataset_yolo_norm_rand_process(self,process_data):
        Count_images = 0
        image_overlap = 0
        process_id=process_data[0]
        list_insect_in_image=process_data[1]
        overlap=process_data[2]
        for index_images,index_group in enumerate(list_insect_in_image):
            Count_images, image_overlap=self.__create_image(Count_images, image_overlap,
                                index_images, index_group, overlap,process_id)
        return [Count_images,image_overlap]
    
    # Create Dataset with labelling for Yolo Random number insect per image
    # def create_dataset_yolov_norm_rand(self, low, high, size, overlap=False):
    #     Count_images = 0
    #     image_overlap = 0
    #     list_insect_in_image = np.random.randint(low=low, high=high, size=size)
    #     for index_images,index_group in enumerate(list_insect_in_image):

            
    #         Count_images, image_overlap=self.__create_image(Count_images, image_overlap,
    #                             index_images, index_group, overlap)
    #     with open(os.path.join(self.path_dataset, f'insects_{len(self.groups_insect_per_image)}_test_new.txt'), 'a') as f:
    #         f.write(
    #             f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')
    #     print(
    #         f'count_of_image={Count_images} count_overlap_image={image_overlap}  overlap {image_overlap/Count_images*100}%')
