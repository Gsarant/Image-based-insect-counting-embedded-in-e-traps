import numpy as np
import cv2
import os
import time
from imutils import paths
import random
from sympy.geometry import Point, Circle, intersection, Polygon


#from create_dataset_funnel.image_insect import Image_Insect
import math

# Create Custom image for Traing and Validating Dataset


class Image_Insects_Funnel:
    def __init__(self, path_funnels, path_dirtes, path_insects):
        if os.path.isdir(path_funnels):
            self.path_funnels = path_funnels
            self.list_funnels = list(paths.list_images(self.path_funnels))
        else:
            raise ValueError(f'{path_funnels} is not path of folder')

        if os.path.isdir(path_dirtes):
            self.path_dirtes = path_dirtes
            self.list_dirtes = list(paths.list_images(self.path_dirtes))
        else:
            raise ValueError(f'{path_dirtes} is not path of folder')

        if os.path.isdir(path_insects):
            self.path_insects = path_insects
            self.list_insects = list(paths.list_images(self.path_insects))
        else:
            raise ValueError(f'{path_insects} is not path of folder')
        self.overlap_array = []

    # Return a random Funnel filename from list of funnels
    def __get_random_funnel_name(self):
        full_funnel_name = self.list_funnels[random.randint(
            0, len(self.list_funnels)-1)]
        if os.path.isfile(full_funnel_name):
            return full_funnel_name
        else:
            return None

    # Return the filename of an empty Funnel
    def __get_null_funnel_name(self):
        funnel_name = 'ICUIMG-f_20220604193550_0.jpg'
        full_funnel_name = os.path.join(self.path_funnels, funnel_name)
        if os.path.isfile(full_funnel_name):
            return full_funnel_name
        else:
            return None

    # Return a random insect filename from list of insects
    def __get_random_insect_name(self):
        full_insect_name = self.list_insects[random.randint(
            0, len(self.list_insects)-1)]
        while full_insect_name.endswith('jpg'):
            if os.path.isfile(full_insect_name):
                return full_insect_name
            else:
                return None

    # Return a random dirty of list of dirtes
    def __get_random_dirtes_name(self):
        full_dirtes_name = self.list_dirtes[random.randint(
            0, len(self.list_dirtes)-1)]
        while full_dirtes_name.endswith('jpg'):
            if os.path.isfile(full_dirtes_name):
                return full_dirtes_name
            else:
                return None

    # Return an image with random funnel
    def __create_background(self, null_funnel=False, grayscale=False):
        if null_funnel:
            image_funnel = cv2.imread(self.__get_null_funnel_name())
        else:
            image_funnel = cv2.imread(self.__get_random_funnel_name())
        if grayscale:
            image_funnel = cv2.cvtColor(image_funnel, cv2.COLOR_BGR2GRAY)
        return image_funnel

    # Return an image with random brigthness for augmentation
    def __brightness_image(self, image, aply_to_all_image=False):
        brightness = random.randint(-100, 100)
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness

            al_pha = (max - shadow) / 255
            ga_mma = shadow

            image = cv2.addWeighted(image, al_pha,
                                    image, 0, ga_mma)
        return image

    # Return an image with random contrast for augmentation
    def __contrast_image(self, image, aply_to_all_image=False):
        contrast = random.randint(-80, 80)
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            image = cv2.addWeighted(image, Alpha,
                                    image, 0, Gamma)
        return image

    # Return an image with random rotation for augmentation
    def __rotate_image(self, image, angle=0, pad=9):
        dim = (max(image.shape[:2])+pad, max(image.shape[:2])+pad, 3)
        new_image = np.full(dim, 255, dtype=np.uint8)
        y0 = int(.5 * new_image.shape[0]) - int(.5 * image.shape[0])
        x0 = int(.5 * new_image.shape[1]) - int(.5 * image.shape[1])
        y1 = int(.5 * new_image.shape[0]) + int(.5 * image.shape[0])
        x1 = int(.5 * new_image.shape[1]) + int(.5 * image.shape[1])
        if x1-x0 < image.shape[1]:
            x1 = image.shape[1]+x0
        if y1-y0 < image.shape[0]:
            y1 = image.shape[0]+y0
        new_image[y0:y1, x0:x1, :] = image
        height, width = new_image.shape[:2]
        center = (int(width/2), int(height/2))
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=random.randint(1, 359) if angle == 0 else angle, scale=1)
        rot_im = cv2.warpAffine(src=new_image, M=rotate_matrix, dsize=(
            width, height), borderValue=(255, 255, 255))
        return rot_im

    # Return an image with vertical flip for augmentation
    def __flip_vertical_image(self, image):
        return cv2.flip(image, 0)

    # Return an image with horizontal flip for augmentation
    def __flip_horizontal_image(self, image):
        return cv2.flip(image, 1)

    # Check if a object is overlaped with other objects in custom image
    # def __is_it_overlap(self, image):
    #    for item_overlap_array in self.overlap_array:
    #        if any(e in range(item_overlap_array[0], item_overlap_array[1]) for e in range(image[0], image[1])) and any(e in range(item_overlap_array[2], item_overlap_array[3]) for e in range(image[2], image[3])):
    #            return True
    #    return False
    def __is_it_overlap(self, rect_array, rect_image):
        for rect in rect_array:
            rect_points=rect.vertices
            rect_image_points=rect_image.vertices
            if any(e in range(rect_points[0][0], rect_points[2][0]) for e in range(rect_image_points[0][0], rect_image_points[2][0]))\
                and any(e in range(rect_points[0][1], rect_points[2][1]) for e in range(rect_image_points[0][1], rect_image_points[2][1])):
                return True
            #for rect_image_point in rect_image.vertices:
            #    if rect.encloses_point(rect_image_point):
            #        return True
        return False
    # Return an scaled image  for augmentation

    def __scale_image(self, image):
        scale = np.random.uniform(low=0.8, high=1.1, size=1)
        image = cv2.resize(
            image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), cv2.INTER_AREA)
        return image

    # Paste an object to backround image
    def __paste(self, background_part, pasted_image):
        pasted_image_bg = np.full(pasted_image.shape, 255)
        pasted_image_bg = cv2.cvtColor(pasted_image, cv2.COLOR_RGB2GRAY)
        #pasted_image_bg = cv2.cvtColor(pasted_image_bg, cv2.COLOR_GRAY2RGB)
        #background_part = np.where( pasted_image_bg < 240, pasted_image, background_part)
        background_part[:,:,0] = np.where( pasted_image_bg < 240, pasted_image[:,:,0], background_part[:,:,0])
        background_part[:,:,1] = np.where( pasted_image_bg < 240, pasted_image[:,:,1], background_part[:,:,1])
        background_part[:,:,2] = np.where( pasted_image_bg < 240, pasted_image[:,:,2], background_part[:,:,2])
            #150
        return background_part

    def __group_insects(self, funnel_circle, group_circles, image_coords):

        def is_overlap(outcircle, incircle):
            points = incircle.bounds
            if outcircle.encloses_point(Point(int(points[0]),int(points[1]))):
                return True
            if outcircle.encloses_point(Point(int(points[0]),int(points[3]))):
                return True
            if outcircle.encloses_point(Point(int(points[2]),int(points[1]))):
                return True
            if outcircle.encloses_point(Point(int(points[2]),int(points[3]))):
                return True
            return False

        def circle_to_rect(circle, width, heigth):
            x1 = int(circle.center.x-width/2)
            x2 = int(circle.center.x+width/2)
            y1 = int(circle.center.y-heigth/2)
            y2 = int(circle.center.y+heigth/2)
            return x1, y1, x2, y2

        def is_point_in__circle(circles, point_c1_c2):
            for p in circles:
                if p.encloses_point(point_c1_c2):
                    return True
            return False

        
        x1, y1, x2, y2 = image_coords
        image_width = x2-x1
        image_height = y2-y1
        new_circle_r = int(min(image_width,image_height)/3)
        new_circle = None
        #first insect image create a circle to group
        if not group_circles is None and len(group_circles) == 0:
            new_circle = Circle(
                (int(x1+image_width/2), int(y1+image_height/2)), int(new_circle_r))
            if  not is_overlap(funnel_circle, new_circle):
                new_circle = None
        elif not group_circles is None and len(group_circles) == 1:
            #Second circle is a random circle around first circle
            while True:
                theta = np.random.uniform(
                    low=0, high=2*np.pi, size=1).item(0)  # angle
                periferal_x = int(
                    group_circles[0].radius * np.cos(theta)+group_circles[0].center.x)
                periferal_y = int(
                    group_circles[0].radius * np.sin(theta) + group_circles[0].center.y)
                new_point_center_x = int(
                    new_circle_r * np.cos(theta) + periferal_x)
                new_point_center_y = int(
                    new_circle_r * np.sin(theta) + periferal_y)

                x1_background = int(new_point_center_x-image_width/2)
                y1_background = int(new_point_center_y-image_height/2)
                y2_background = y1_background+image_width
                x2_background = x1_background+image_height
                new_circle = Circle((int(x1_background+image_width/2),
                                    int(y1_background+image_height/2)), int(new_circle_r))
                if len(intersection(new_circle, group_circles[0])) == 0:
                    if  is_overlap(funnel_circle, new_circle):
                        break
        else:
            #all oithers circles is around of last two circles
            Circle1 = Circle((int(group_circles[-1].center.x), int(
                group_circles[-1].center.y)), int(group_circles[-1].radius+new_circle_r))

            Circle2 = Circle((int(group_circles[-2].center.x), int(
                group_circles[-2].center.y)), int(group_circles[-2].radius+new_circle_r))

            point_c1_c2 = intersection(Circle1, Circle2)
            if len(point_c1_c2) > 1:
                new_circle = Circle(point_c1_c2[0], new_circle_r)
                x1, y1, x2, y2 = circle_to_rect(new_circle, image_width, image_height)
                if x1>0 and x2>0 and y1>0 and y2>0:
                    if is_point_in__circle(group_circles[-5:], point_c1_c2[0]):
                        new_circle = Circle(point_c1_c2[1], new_circle_r)
                        x1, y1, x2, y2 = circle_to_rect(new_circle, image_width, image_height)
                        if x1>0 and x2>0 and y1>0 and y2>0:
                            
                            if  is_overlap(funnel_circle,new_circle):
                                new_circle = Circle(point_c1_c2[0], new_circle_r)
                            else:
                                new_circle = Circle(point_c1_c2[0], new_circle_r)    
                        else:
                            new_circle = Circle(point_c1_c2[0], new_circle_r)
                    else:
                        new_circle = Circle(point_c1_c2[0], new_circle_r)
                else:
                    new_circle = Circle(point_c1_c2[1], new_circle_r)
        if not new_circle is None:
            if  self.point_in_base_trap(new_circle.center.x,new_circle.center.y):

                group_circles.append(new_circle)
                x1, y1, x2, y2 = circle_to_rect(new_circle, image_width, image_height)
                while x2-x1 != image_width:
                    x2 += 1
                while y2-y1 != image_height:
                    y2 += 1
                return x1, y1, x2, y2
            else:
                return None, None, None, None
        else:
            return None, None, None, None
    
    def point_in_base_trap(self,x,y):
        point=Point(x,y)
        base=Circle(Point(860,440), 360)
        return  base.encloses_point(point)
    
    
    # Propose a random position in background image , check for overlap and paste object to backgrounfd image
    # Return Background image with pasted object, Value if object is overlaped with an other object, position of object in background image
    def __paste_image_area(self, background, pasted_image, rect_array=[], groups_array=[], group_circles=[], overlap=False):
        count = 0
       # overlap = False
        #if  len(background.shape)>2 and background.shape[2]>3:
        #    pass
        while True:
            # Propose Random position
            r = np.random.uniform(low=0, high=1, size=1).item(0)  # radius
            theta = np.random.uniform(
                low=0, high=2*np.pi, size=1).item(0)  # angle

            circle_r = 360
            x1_background = int(circle_r*np.sqrt(r) * np.cos(theta) + 860)
            y1_background = int(circle_r*np.sqrt(r) * np.sin(theta) + 440)

            y2_background = y1_background+pasted_image.shape[0]
            x2_background = x1_background+pasted_image.shape[1]

            # make groups of insect
            if not groups_array is None and len(groups_array) > 0:
                x1_background, y1_background, x2_background, y2_background = self.__group_insects(Circle(Point(860, 440), int(circle_r)),
                                                                                                  group_circles,
                                                                                                  (x1_background, y1_background, x2_background, y2_background))
                #if not x1_background is None and not y1_background is None and not x2_background is None and not y2_background is None :
                if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                    rect = Polygon(Point(int(x1_background), int(y1_background)),
                                Point(int(x2_background),
                                        int(y1_background)),
                                Point(int(x2_background),
                                        int(y2_background)),
                                Point(int(x1_background),
                                        int(y2_background)),
                                )
                    #if x2_background>0 and x1_background>0 and y2_background>0 and y1_background>0:
                # if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                    if not groups_array is None and len(groups_array) > 0:
                        groups_array[0] -= 1
                        if groups_array[0] == 0:
                            groups_array.remove(0)
                            group_circles.clear()
                    break
                else:
                        groups_array.pop()
                        group_circles.clear()
            else:
                if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                    # Check overlap
                    rect = Polygon(Point(int(x1_background), int(y1_background)),
                                Point(int(x2_background), int(y1_background)),
                                Point(int(x2_background), int(y2_background)),
                                Point(int(x1_background), int(y2_background)),
                                )
                    if overlap == False:
                        if not self.__is_it_overlap(rect_array, rect):
                            count = 0
                            #if x2_background>0 and x1_background>0 and y2_background>0 and y1_background>0:
                            #if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                            break
                        else:
                            count = count+1
                            if count > 20:
                                overlap = True
                                #if x2_background>0 and x1_background>0 and y2_background>0 and y1_background>0:
                            #    if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                                break
                    else:
                        #overlap = True
                        #if x2_background>0 and x1_background>0 and y2_background>0 and y1_background>0:
                        #if background[y1_background:y2_background, x1_background:x2_background, :].shape[1]  ==pasted_image.shape[1] and background[y1_background:y2_background, x1_background:x2_background, :].shape[0]  ==pasted_image.shape[0] :
                        break
        
        background[y1_background:y2_background, x1_background:x2_background, :] = self.__paste(
            background[y1_background:y2_background, x1_background:x2_background, :], pasted_image)
        return background, overlap, rect

    # Load an image from a file
    def __load_image(self, path_image):
        try:
            image = cv2.imread(path_image)
            return image
        except:
            return None

    # Paste random number of dirtes in Background image
    # Return Background image with pasted dirtes,  position of dirtes in background image
    def __put_dirtes_to_background(self, funnel, max_num_of_dirtes=6):
        dirtes_points = []

        for index in range(random.randint(0, max_num_of_dirtes)):
            dirty_name = self.__get_random_dirtes_name()
            dirty_image = image = self.__load_image(dirty_name)
            img = dirty_image.copy()
            if random.random() > 0.01:
                img = self.__rotate_image(dirty_image, 90)
            if random.random() > 0.5:
                img = self.__flip_horizontal_image(dirty_image)
            if random.random() > 0.5:
                img = self.__flip_vertical_image(dirty_image)
            if random.random() > 0.8:
                img = self.__scale_image(dirty_image)
            # make dirty window smaller
            x0 = 0
            while np.min(img[x0, :]) > 200:
                x0 += 1
            y0 = 0
            while np.min(img[:, y0]) > 200:
                y0 += 1
            x1 = img.shape[0]-1
            while np.min(img[x1, :]) > 200:
                x1 -= 1
            y1 = img.shape[1]-1
            while np.min(img[:, y1]) > 200:
                y1 -= 1
            funnel, overlay, rect = self.__paste_image_area(
                funnel, img[x0:x1, y0:y1], [], [], [], False)
            dirtes_points.append((rect.vertices[0].x,
                                  rect.vertices[0].y,
                                  rect.vertices[2].x,
                                  rect.vertices[2].y))

        return funnel, dirtes_points

    # Create Custom image with random funnel, random dirtes and a specific number of insects
    def create_image(self, insects_num, insects_overlap=False):
        funnel = self.__create_background(True)
        funnel, dirtes_points = self.__put_dirtes_to_background(funnel,2)
        count_overlap = 0
        rect_array = []
        points_array = []
        groups_array=[]
        max_num_groups=int(insects_num*0.4)
        for group_num in range(random.randint(0,3)):
           if max_num_groups>2:
               a=random.randint(2,max_num_groups)
               groups_array.append(a)
               max_num_groups=max_num_groups-a
        groups_array.append(max_num_groups)
        group_circles = []
        for index_insects in range(insects_num):
            while True:
                insect_name = self.__get_random_insect_name()
                image = self.__load_image(insect_name)
                
                if image is not None:
                    img = image
                    if random.random() > 0.01:
                        img = self.__rotate_image(image, 90)
                    if random.random() > 0.5:
                        img = self.__flip_horizontal_image(image)
                    if random.random() > 0.5:
                        img = self.__flip_vertical_image(image)
                    if random.random() > 0.8:
                        img = self.__scale_image(image)
                    # make insect window smaller
                    x0 = 0
                    while np.min(img[x0, :]) > 200:
                        x0 += 1
                    y0 = 0
                    while np.min(img[:, y0]) > 200:
                        y0 += 1
                    x1 = img.shape[0]-1
                    while np.min(img[x1, :]) > 200:
                        x1 -= 1
                    y1 = img.shape[1]-1
                    while np.min(img[:, y1]) > 200:
                        y1 -= 1
                    funnel, overlap, rect = self.__paste_image_area(funnel,
                                                                    img[x0:x1,
                                                                        y0:y1],
                                                                    rect_array,
                                                                    groups_array,
                                                                    group_circles,
                                                                    insects_overlap)
                    rect_array.append(rect)

                    points_array.append((rect.vertices[0].x,
                                        rect.vertices[0].y,
                                        rect.vertices[2].x,
                                        rect.vertices[2].y))
                    if overlap == True:
                        count_overlap = count_overlap+1
                    break
        if insects_num==0:
            p=0.5
        else:
            p=0.8
        if random.random() > p :
            funnel = self.__brightness_image(funnel, True)
        elif random.random() > p:
            funnel2 = self.__contrast_image(funnel, True)
        return funnel, count_overlap, points_array, dirtes_points
