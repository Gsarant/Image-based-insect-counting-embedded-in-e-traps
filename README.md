# Image-based insect counting embedded in e-traps
## Image-based insect counting embedded in e-traps that learn without manual image annotation and self-dispose captured insects

###  Raspberry Pi 4 B Models Results
#### Overlap
| Model Name | pa | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.4 IOU 0.4 | 0.71 | 0.86 | 543.7 |
| Yolov8 CONF 0.3 IOU 0.8 | 0.76 | 0.47 | 645.0 |
| CSRNet_HVGA | 0.88 | 0.22 | 6106.8 |
| Count_Regression_resnet18 | 0.72 | 0.54 | 383.5 |
| Count_Regression_resnet50 | 0.93 | 0.13 | 717.2 |

#### Helicoverpa armigera 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7_Helicoverpal CONF 0.3 IOU 0.5 | 0.69 | 4.71 | 535.4|
|Yolov8_Helicoverpal CONF 0.3 IOU 0.4|0.72|4.08|615.9|
|CSRNet_Helicoverpa_HVGA|0.71|4.12|6327.1|
|Count_Regression_Helicoverpa_resnet18|0.78|2.89|381.5|
|Count_Regression_Helicoverpa_resnet50|0.69|4.35|699.5|

#### Plodia interpunctella 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov7 CONF 0.3 IOU 0.8|0.61|3.00|548.4|
|Yolov8 CONF 0.3 IOU 0.85|0.51|4.07|604.3|
|CSRNet_HVGA|0.63|2.27|6229.0|
|Count_Regression_resnet18|0.33|8.19|374.4|
|Count_Regression_resnet50|0.48|5.12|699.2|

#### Helicoverpa armigera 50
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov8 CONF 03 IOU 0.4|0.77|20.39|624.3|
|CSRNet_HVGA|0.88|6.03|6312.9|
|Count_Regression_resnet18|0.37|31.29|381.5|
|Count_Regression_resnet50|0.72|13.82|717.2|

#### Plodia interpunctella 50-100
 Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov7 CONF 0.3 IOU 0.8|0.87|9.83|543.3|
|Yolov8 CONF 0.3 IOU 0.85|0.69|26.52|659.8|
|CSRNet _HVGA|0.76|21.26|6300.9|
|Count_Regression_resnet18|0.27|61.59|380.4|
|Count_Regression_resnet50|0.36|55.56|698.2|


###  Raspberry Pi Zero Models Results
#### Helicoverpa armigera 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.4 | 0.65 | 5.42 | 32540.3 |
| CSRNet _HVGA medium |0.0049 | 14.85 | 7267.4 |
| CSRNet _HVGA quantized | 0.32 | 10.18 | 28337.9 |
| Count_Regression_resnet18 | 0.78 | 2.89 | 1682.6 |
| Count_Regression_resnet50 | 0.69 | 4.35 | 3201.8 |


#### Plodia interpunctella 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.8 | 0.58 | 3.64 | 3183.8 |
| CSRNet_HVGA_medium | 0.57 | 3.72 | 7257.0 |
| CSRNet_HVGA quantized | 0.64 |2.42 |28520.0 |
| Count_Regression_resnet18 | 0.33 | 8.19 | 1674.5 |
| Count_Regression_resnet50 | 0.48 | 5.12 | 3139.8 |

#### Helicoverpa armigera 50
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.4 | 0.23 | 38.20 | 3256.5 |
| CSRNet_HVGA quantized | 0.27 | 36.46 | 28339.3 |
| Count_Regression_resnet18 | 0.37 | 31.29 |1676.6 |
| Count_Regression_resnet50 | 0.72 | 13.82 |3161.4 |

#### Plodia interpunctella 50-100
 Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.8 | 0.84 | 12.30 |3226.1|
| CSRNet_HVGA_medium | 0.62 | 32.62 | 7257.7 |
|CSRNet_HVGA quantized|0.88|10.24|28442.0|
|Count_Regression_resnet18|0.27|61.59|1757.1|
|Count_Regression_resnet50|0.36|55.56|3446.7|


###Links
* [Empty Fannel images](https://drive.google.com/file/d/1-N29XwUH_AyjFeHnz69N9klgvkQYZpk-/view?usp=sharing)
* [Helicoverpa original fixed images](https://drive.google.com/file/d/1-KV_UFt07zjjtZxBmG_vO4QTxoaULq-6/view?usp=sharing)
* [Plodia original fixed images](https://drive.google.com/file/d/1-KXeEdTuEOycDd9MRC8r4csz-Ox9RmEU/view?usp=sharing)
* [Helicoverpa croped insects images](https://drive.google.com/file/d/1-N29XwUH_AyjFeHnz69N9klgvkQYZpk-/view?usp=sharing)
* [Plodia croped insects images](https://drive.google.com/file/d/1-RmWV4w4dd30gTQnCOyi0O98fiqt0Fys/view?usp=share_link)
* [Helicoverpa dataset images](https://drive.google.com/file/d/1aWR88TkmgFx1P3M4xPtRFR5hBrM0jBR8/view?usp=sharing)
* [Plodia dataset images](https://drive.google.com/file/d/1vK1oZkMkCG_Q0vFTzUhovSYLUn5AfiWB/view?usp=sharing)
* [Count Regression Pytorch Models](https://drive.google.com/drive/folders/134VitRY9ZOnTqoVMZfZ6PdsAaeDh6u2g?usp=sharing) 
* [CSRnet Crowd Counting Pytorch Models](https://drive.google.com/drive/folders/1re1IpdehEwzmgX9t_C8_1O-UYy2yvOkY?usp=sharing) 
* [Yolov7 Pytorch Models](https://drive.google.com/drive/folders/1hOB2hYBwvxH8NEi7xiOg9yb2qTDQyt8q?usp=sharing) 
* [Yolov7 Pytorch Models](https://drive.google.com/drive/folders/1w1LrFaE3vNOqbHM6lNUhsc7hN-VIg_w0?usp=sharing) 
* [Count Regression TFLite Models](https://drive.google.com/drive/folders/1haB8WX-D-5mGYq6E6l7uBfTSBImS8Mkd?usp=sharing) 
* [CSRnet Crowd Counting TFLite Models](https://drive.google.com/drive/folders/1A3Gmw1qXuEwgkAE588GffHyTNxAwZW-W?usp=share_link) 
* [Yolov7 TFLite Models](https://drive.google.com/drive/folders/1AgB4hLVkeXUeKFLkGIDcwdVJwBIWLPzn?usp=share_link) 
* [Test_set](https://drive.google.com/file/d/1-cLyBnZ7vjNUK5f0OfGP6QFNwaJo5KXI/view?usp=share_link)
* [journal]()

