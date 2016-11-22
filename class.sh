set -e

make all -j4

MODEL="./models/hed/deploy.prototxt"
WEIGHTS="./models/hed/hed_pretrained_bsds.caffemodel"
MEAN="104.00698793,116.66876762,122.67891434"
#MEAN="122.67891434,116.66876762,104.00698793"
#INPUT=$@
#INPUT="../Seq05VD_f01740.png"
INPUT="../100039.jpg"
#INPUT="../File-Irish-terrier.jpg"
echo $WEIGHTS
build/tools/classification.bin -model $MODEL -weights $WEIGHTS -mean $MEAN -output fuse.png $INPUT


