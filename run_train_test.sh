if [ -z ${WIN10_IP} ]; then
  echo "at first, please define the environmental variable WIN10_IP"
  exit
fi

CONFIG=$1
WORKDIR=/media/ubuntu/Temp/mmdetection/$CONFIG

#./tools/dist_train.sh configs/faster_rcnn_gd_1024_4classes/$CONFIG.py 2 \
#  --work-dir ${WORKDIR} || exit

## LOG_JSON_FILE=`ls -alt ${WORKDIR}/*.log.json | cut -f10- -d" "`     #  this is not good !!!
LOG_JSON_FILE=`ls -alt ${WORKDIR}/*.log.json | head -n 1 | grep -oE '[^ ]+$'`
LOG_FILE="${LOG_JSON_FILE%.*}"
echo $LOG_JSON_FILE
echo $LOG_FILE

python ./tools/analysis_tools/analyze_logs.py plot_curve \
  ${LOG_JSON_FILE} --out ${WORKDIR}/log_curve.png || exit

LOG_CURVE_FILE=`ls -alt ${WORKDIR}/log_curve*.png | head -n 1 | grep -oE '[^ ]+$'`
echo $LOG_CURVE_FILE
filename=$(basename -- "$LOG_CURVE_FILE")
extension="${filename##*.}"
filename="${filename%.*}"
epoch_num=${filename##*_}
echo $filename
echo $extension
echo $epoch_num
CKPT_FILE=$WORKDIR/epoch_${epoch_num}.pth
echo $WORKDIR/epoch_${epoch_num}.pth

# the following is optional
#python tools/test.py $WORKDIR/$CONFIG.py $CKPT_FILE \
#  --show-dir ${WORKDIR}/epoch_${epoch_num}_val || exit

# faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug2
ssh ${WIN10_IP} powershell -c mkdir E:/mmdetection/work_dirs/${CONFIG};
scp $LOG_FILE $LOG_JSON_FILE $LOG_CURVE_FILE $CKPT_FILE $WORKDIR/$CONFIG.py \
${WIN10_IP}:E:/mmdetection/work_dirs/${CONFIG}







