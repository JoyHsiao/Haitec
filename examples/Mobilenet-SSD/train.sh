#!/bin/sh
if ! test -f example/8_class_haitech_prototxt/MobileNetSSD_train.prototxt ;then
	echo "error: example/8_class_haitech_prototxt/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
../../build/tools/caffe train -solver="solver_train.prototxt" -gpu 0 -weights="snapshot_mobilenet_8classes/mobilenet_iter_107000.caffemodel"
#-snapshot="/home/ubuntu/caffe/examples/MobileNet-SSD/snapshot_mobilenet_8classes/mobilenet_iter_107000.solverstate"






