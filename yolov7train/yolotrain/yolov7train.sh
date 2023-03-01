#! /bin/bash
cd /home/giannis/yolov7/ \
&& \

python train.py --weights /home/giannis/paper2/yolotrain/yolov7-tiny.pt \
                --data /home/giannis/paper2/yolotrain/myyalm/helicoverpa.yaml \
                --workers 32 --batch-size 64 --img 480 320 --epochs 30 --device 0 \
                --cfg cfg/training/yolov7-tiny.yaml \
                --name /home/giannis/paper2/yolotrain/Yolov7runs_10k/Helicoverpa-armigera \
                --hyp data/hyp.scratch.tiny.yaml \
#&& \
#python train.py --weights /home/giannis/paper2/yolotrain/yolov7-tiny.pt \
#                --data /home/giannis/paper2/yolotrain/myyalm/plodia.yaml \
#                --workers 4 --batch-size 64 --img 480 320 --epochs 300 --device 0 \
#                --cfg cfg/training/yolov7-tiny.yaml \
#                --name /home/giannis/paper2/yolotrain/Yolov7runs_10k/Plodia \
#                --hyp data/hyp.scratch.tiny.yaml \