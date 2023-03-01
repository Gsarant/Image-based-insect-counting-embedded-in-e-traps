# Create Dataset Funnel
## Introduction
We create synthetic dataset for deep learning trainnig.


## Training and Validation Dataset.
The Training sets, Validation sets and annotations were generated with the following method.

### Steps
1. We took some photos from an empty funnel trap or one with specks of dirt in it (set of image [class0](https://drive.google.com/file/d/1-47kKdPrK8Yyw5ZPl7l4keTz6_KuHFL2/view?usp=share_link) ).
2. We put in and posed an insect, then took a photo with raspberry pi zero. 
3. This step is repeated with insects of both types.
We were left with two sets of images (one for each type of insect)  with a funnel trap as background and said insect in a specific position. (Helicoverpa armigera images set [class_1_fixed.tar.gz](https://drive.google.com/file/d/1-KV_UFt07zjjtZxBmG_vO4QTxoaULq-6/view?usp=share_link) and Plodia interpunctella images set [class_2_fixed.tar.gz](https://drive.google.com/file/d/1-KXeEdTuEOycDd9MRC8r4csz-Ox9RmEU/view?usp=share_link) )

We passed  these image sets  from create_insect_dataset_rembg.py and cropped the posed insect from the funnel trap. This leaves us with two sets of images; one with natural lighting and the insects in different body posture. ([Helicoverpa_armigera_insects_new.tar.gz](https://drive.google.com/file/d/1-N29XwUH_AyjFeHnz69N9klgvkQYZpk-/view?usp=share_link) and [Plodia_interpunctella_insects_new.tar.gz](https://drive.google.com/file/d/1-RmWV4w4dd30gTQnCOyi0O98fiqt0Fys/view?usp=share_link)

### create_insect_dataset_rembg.py
This program crops the posed insect, removes the background , and passes the image through a number of filters. it then saves the insect in an image with a white background. This way two sets of images are created, one set for Helicoverpa armigera and another one for Plodia interpunctella.

### create_dataset_funnel
This program creates new images, from empty funnel trap images [class0](https://drive.google.com/file/d/1-47kKdPrK8Yyw5ZPl7l4keTz6_KuHFL2/view?usp=share_link) and the images with cropped insects from the steps below. With the method presented, one can create thousands of images that look like natural images from the funnel trap.

#### Steps
1. Take an image of an empty funnel trap.
2. Put in a random position insects,in a random pose, one random image from insects sets from the steps below.
3. Create an annotation for this insect and continue. The annotation style and folder hierarchy are meant for the yolo training model.
4. For data augmentation the program changes the lighting, the color contrast, while also rotating the insects horizontally or by 45 degrees. All these changes are made at a rate of 20%.
5. Occasionally, add more specks of dirt. [dirtes.tar.gz](https://drive.google.com/file/d/1-WiVJMtZjTjux1d7ssDGvB8cqx5QFAfW/view?usp=share_link)
#### Groups of insects
For more natural rendering 40% of image insects create (1-2) neighbor groups of insects that are too close to each other or overlap one part of the other.
![first two insects in neighbor groups](images/2insect.jpg)
![add more insects to neighbor groups](images/3d4dinsect.jpg)


###Datasets 
 * Helicoverpa armigera insects ( [datasets_Helicoverpa_armigera_10k.tar.xz](https://drive.google.com/file/d/1aWR88TkmgFx1P3M4xPtRFR5hBrM0jBR8/view?usp=share_link)), 
* Plodia interpunctella insects ( [datasets_Plodia_interpunctella_10k.tar.xz](https://drive.google.com/file/d/1vK1oZkMkCG_Q0vFTzUhovSYLUn5AfiWB/view?usp=share_link) ). Every dataset has 10.000  images for the Train set and 500 for the Validation set.













