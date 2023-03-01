#! /bin/bash
cd /home/giannis/yolov7/ \
&& \
python export.py --weights '/home/giannis/paper2/yolotrain/Yolov7runs_10k/Helicoverpa-armigera/weights/best.pt' \
        --grid --simplify \
        --topk-all 200 \
        --img-size 480 320  --max-wh 480 \
&& \
python export.py --weights '/home/giannis/paper2/yolotrain/Yolov7runs_10k/Plodia/weights/best.pt' \
        --grid --simplify \
        --topk-all 200 \
        --img-size 480 320  --max-wh 480 \

