import os
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter 
from scipy import spatial
import scipy
import glob 

class Create_GrowndTruth_Image():
        def __init__(self, image_dirname,
                     density_dirname,
                     Input_Image_resize=None,
                     scale_image=1,
                     ):
                self.image_dirname=image_dirname
                self.density_dirname=density_dirname
                self.scale_image=scale_image
                self.Input_Image_resize=Input_Image_resize
                
        #this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
        def gaussian_filter_density(self,gt):
            density = np.zeros(gt.shape, dtype=np.float32)
            gt_count = np.count_nonzero(gt)
            if gt_count == 0:
                return density
        
            pts = np.asanyarray(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
            leafsize = 2048
            # build kdtree
            tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
            # query kdtree
            distances, locations = tree.query(pts, k=4)

            for i, pt in enumerate(pts):
                pt2d = np.zeros(gt.shape, dtype=np.float32)
                pt2d[pt[1],pt[0]] = 1.
                if gt_count > 1:
                    sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
                else:
                    sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
                density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
            return density

        def generate_density_map_with_fixed_kernel(self,img_size_h_w,points,kernel_size=15,sigma=4.0):
        #        img: input image.
        #  points: annotated pedestrian's position like [row,col]
        #  kernel_size: the fixed size of gaussian kernel, must be odd number.
        #  sigma: the sigma of gaussian kernel.
        #   return:
        #   d_map: density-map we want
        
            def guassian_kernel(img_size,sigma):
                rows=img_size[0] # mind that size must be odd number.
                cols=img_size[1]
                mean_x=int((cols-1)/2)
                mean_y=int((rows-1)/2)

                f=np.zeros(img_size)
                for x in range(0,cols):
                    for y in range(0,rows):
                        mean_x2=(x-mean_x)*(x-mean_x)
                        mean_y2=(y-mean_y)*(y-mean_y)
                        f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
                return f

            rows,cols=img_size_h_w
            d_map=np.zeros([rows,cols])
            f=guassian_kernel([kernel_size,kernel_size],sigma) # generate gaussian kernel with fixed size.
            normed_f=(1.0/f.sum())*f # normalization for each head.

            if len(points)==0:
                return d_map
            else:
                for p in points:
                    r,c=p[0],p[1]
                    if r>=rows or c>=cols:
                        continue
                    for x in range(0,f.shape[0]):
                        for y in range(0,f.shape[1]):
                            if x+((r+1)-int((f.shape[0]-1)/2))<0 or x+((r+1)-int((f.shape[0]-1)/2))>rows-1 \
                            or y+((c+1)-int((f.shape[1]-1)/2))<0 or y+((c+1)-int((f.shape[1]-1)/2))>cols-1:
                                continue
                            else:
                                d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))]+=normed_f[x,y]
            return d_map
      
        
        def create_growndtruth_from_image(self,
                                          image_name,
                                          density_dirname,
                                          kernel_size=15,
                                          sigma=4.0
                                          ):
                image=None
                if  self.Input_Image_resize is None:
                    image=cv2.imread(image_name)
                    img_size_w_h=(image.shape[1],image.shape[0])
                else:   
                    img_size_w_h= self.Input_Image_resize
                   
                
                density_name = image_name.split("/")[-1].replace(".jpg", ".txt")
                density_name=os.path.join(density_dirname,density_name)
              
                points=[]
                with open(density_name,'r') as f:
                    for count,line in enumerate(f):
                        obj,x,y,w,h,_=line.split('  ')
                        if obj!='0':
                            points.append([int(float(y)*img_size_w_h[1]),int(float(x)*img_size_w_h[0])])
                 
                image_row_col=(img_size_w_h[1],img_size_w_h[0])
                return self.generate_density_map_with_fixed_kernel(image_row_col, points, kernel_size, sigma)
                