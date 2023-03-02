
# Insect Count Methods in Rasbperry Pi 4B
##  Models Results
### Overlap
| Model Name | pa | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.4 IOU 0.4 | 0.71 | 0.86 | 543.7 |
| Yolov8 CONF 0.3 IOU 0.8 | 0.76 | 0.47 | 645.0 |
| CSRNet_HVGA | 0.88 | 0.22 | 6106.8 |
| Count_Regression_resnet18 | 0.72 | 0.54 | 383.5 |
| Count_Regression_resnet50 | 0.93 | 0.13 | 717.2 |

### Helicoverpa armigera 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7_Helicoverpal CONF 0.3 IOU 0.5 | 0.69 | 4.71 | 535.4|
|Yolov8_Helicoverpal CONF 0.3 IOU 0.4|0.72|4.08|615.9|
|CSRNet_Helicoverpa_HVGA|0.71|4.12|6327.1|
|Count_Regression_Helicoverpa_resnet18|0.78|2.89|381.5|
|Count_Regression_Helicoverpa_resnet50|0.69|4.35|699.5|


### Plodia interpunctella 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov7 CONF 0.3 IOU 0.8|0.61|3.00|548.4|
|Yolov8 CONF 0.3 IOU 0.85|0.51|4.07|604.3|
|CSRNet_HVGA|0.63|2.27|6229.0|
|Count_Regression_resnet18|0.33|8.19|374.4|
|Count_Regression_resnet50|0.48|5.12|699.2|


### Helicoverpa armigera 50
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov8 CONF 03 IOU 0.4|0.77|20.39|624.3|
|CSRNet_HVGA|0.88|6.03|6312.9|
|Count_Regression_resnet18|0.37|31.29|381.5|
|Count_Regression_resnet50|0.72|13.82|717.2|


### Plodia interpunctella 50-100
 Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
|Yolov7 CONF 0.3 IOU 0.8|0.87|9.83|543.3|
|Yolov8 CONF 0.3 IOU 0.85|0.69|26.52|659.8|
|CSRNet _HVGA|0.76|21.26|6300.9|
|Count_Regression_resnet18|0.27|61.59|380.4|
|Count_Regression_resnet50|0.36|55.56|698.2|
