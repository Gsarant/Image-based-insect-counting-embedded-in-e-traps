# Create Dataset Funnel
## Introduction
We create synthetic dataset for deep learning trainnig.


## Construct Training and Validation Dataset.
We first collect agricultural pests with typical funnel traps from the field and we subsequently terminate the pests by freezing them. We then carefully position each insect in a preselected, marked location in the e-funnel’s bucket. The angle does not matter as we will rotate the extracted picture later, but we make sure that either the back-wings or the abdomen faces the camera. We then take a picture of the single specimen using the embedded camera that is activated manually by an external bouton. We take one picture per insect (see Fig. 2-left) and we make sure that the training set contains different individuals from the test set. Since we place the insect in a specific location in the backet we can automatically extract from its picture a square containing the insect with almost absolute accuracy as we know beforehand its location (see Fig. 2-bottom). Alternatively, we could perform blob detection and automatically extract the contour of the insect. However, we experimentally found that the former approach is more precise in the presence of shadows. We then remove its background using the python library Rembg (https://github.com/danielgatis/rembg ) that is based on a UNet (see Fig. 2-bottom). This creates a subpicture that follows the contour of the insect closely. Once we have the pictures of the insects, we can proceed in composing the training corpus for all algorithms. A python program selects randomly a picture of an empty bucket that can only have debris that works as a background canvas for the synthesis that places the extracted insect sub-pictures in random locations by uniformly sampling 360 degrees and a radius matching the radius of the bucket (the bottom of the bucket is circular). Besides their random locations, the orientation of each specimen is randomly selected between 0 and 360 degrees before placement, and a uniformly random zoom of ±10% of its size also applies. The number of insects is randomly chosen from a uniform probability distribution between 0-60 for H. armigera and 0-110 for P. interpunctella. We have chosen the upper limit of the distribution by noting that with more than 25 insects of H. armigera, the layering process of insects starts, and image counting becomes by default problematic. Note that, since the e-trap self-disposes the captured insects there is no problem in setting an upper limit other than the power consumption of the rotation process. The upper limit for P. interpunctella is larger because this insect is very small compared to H. armigera and layering, in this case, starts after 100 specimens. Since the program controls the number of insects used to synthesize a picture it also has available their locations and their bounding boxes, and, therefore, can provide the annotated text (i.e., the label) for supervised regressor counters as well as localization algorithms (i.e., YOLO7) and crowd counting approaches. The initial 1664x1232 pixels picture is resized to a resolution of 480x320 pixels for YOLO7 and crowd counting methods to achieve the lowest possible power consumption and storage needs while not affecting the ability of the algorithms to count insects. We synthesized a corpus of 10000 pictures for training and 500 for validation. Starting from the original pictures it takes about 1 sec to construct and fully label (counts and bounding boxes) a synthesized picture. This needs to be contrasted to the time of manual labeling of insects in pictures to see the advantage of our approach.



### Datasets 
 * Helicoverpa armigera insects ( [datasets_Helicoverpa_armigera_10k.tar.xz](https://drive.google.com/file/d/1aWR88TkmgFx1P3M4xPtRFR5hBrM0jBR8/view?usp=share_link)), 
* Plodia interpunctella insects ( [datasets_Plodia_interpunctella_10k.tar.xz](https://drive.google.com/file/d/1vK1oZkMkCG_Q0vFTzUhovSYLUn5AfiWB/view?usp=share_link) ). Every dataset has 10.000  images for the Train set and 500 for the Validation set.













