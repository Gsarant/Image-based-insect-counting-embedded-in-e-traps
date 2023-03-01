#! /bin/bash

while IFS=  read -r -d $'\0'; do
    NEW_EXT=${REPLY/tflite/cc}
    xxd -i $REPLY > $NEW_EXT
    echo "${NEW_EXT} OK"
done < <(find -name *.tflite  -print0 )

#xxd -i /home/giannis/paper2/Insect_CrowdCounting/convert_model/export/CSRNet/Plodia_interpunctella_grey_small_tf/Plodia_interpunctella_grey_small-4.tflite > \
#/home/giannis/paper2/Insect_CrowdCounting/convert_model/export/CSRNet/Plodia_interpunctella_grey_small_tf/Plodia_interpunctella_grey_small-4.cc