
import numpy as np

#I borow from https://github.com/satheeshkatipomu/nms-python/blob/master/NMS%20using%20Python%2C%20Tensorflow.ipynb
def nms_python(bboxes,psocres,score_threshold,iuo_threshold):
    '''
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''
    #Unstacking Bounding Box Coordinates
    psocres=np.array(psocres)
    bboxes=np.array(bboxes)
    filtered = []
    if bboxes is None or len(bboxes)==0:
        return filtered
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    #x_max = bboxes[:,0]+bboxes[:,2]
    #y_max = bboxes[:,1]+bboxes[:,3]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = psocres.argsort()[::-1]
    sorted_idx=sorted_idx[sorted_idx>score_threshold]
    #print(f'  psocres {psocres[sorted_idx]} score_threshold {score_threshold}')
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
    
    #list to keep filtered bboxes.
    
    
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #print(f'rbbox_i {rbbox_i}')
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        #print(f'ious {ious}')
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > iuo_threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #print(f' delete_idx {delete_idx}')
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        #print(f' sorted_idx {sorted_idx}')
        
    
    #Return filtered bboxes
    #return bboxes[filtered].astype('int')
    filtered=np.array(filtered)
    return filtered.astype('int')
