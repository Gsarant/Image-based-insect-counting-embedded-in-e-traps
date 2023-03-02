
# Insect Count Methods in Rasbperry Pi zero
##  Models Results

### Helicoverpa armigera 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.4 | 0.65 | 5.42 | 32540.3 |
| CSRNet _HVGA medium |0.0049 | 14.85 | 7267.4 |
| CSRNet _HVGA quantized | 0.32 | 10.18 | 28337.9 |
| Count_Regression_resnet18 | 0.78 | 2.89 | 1682.6 |
| Count_Regression_resnet50 | 0.69 | 4.35 | 3201.8 |


### Plodia interpunctella 0-20
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.8 | 0.58 | 3.64 | 3183.8 |
| CSRNet_HVGA_medium | 0.57 | 3.72 | 7257.0 |
| CSRNet_HVGA quantized | 0.64 |2.42 |28520.0 |
| Count_Regression_resnet18 | 0.33 | 8.19 | 1674.5 |
| Count_Regression_resnet50 | 0.48 | 5.12 | 3139.8 |

### Helicoverpa armigera 50
| Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.4 | 0.23 | 38.20 | 3256.5 |
| CSRNet_HVGA quantized | 0.27 | 36.46 | 28339.3 |
| Count_Regression_resnet18 | 0.37 | 31.29 |1676.6 |
| Count_Regression_resnet50 | 0.72 | 13.82 |3161.4 |

### Plodia interpunctella 50-100
 Model Name | A | MAE | Time (ms) |
| --- | ---- | --- | --- |
| Yolov7 CONF 0.3 IOU 0.8 | 0.84 | 12.30 |3226.1|
| CSRNet_HVGA_medium | 0.62 | 32.62 | 7257.7 |
|CSRNet_HVGA quantized|0.88|10.24|28442.0|
|Count_Regression_resnet18|0.27|61.59|1757.1|
|Count_Regression_resnet50|0.36|55.56|3446.7|
