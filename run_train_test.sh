if [ -z ${WIN10_IP} ]; then
  echo "at first, please define the environmental variable WIN10_IP"
  exit
fi

CONFIG=$1
WORKDIR=/media/ubuntu/Temp/gd/mmdetection/$CONFIG
WIN10_WORK_ROOT=E:/mmdetection/work_dirs
WIN10_GD_CODE_ROOT=F:/gd
WIN10_GD_DATA_ROOT=F:/gddata/aerial
WIN10_GD_CODE_DRIVE=${WIN10_GD_CODE_ROOT%/*}

CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/faster_rcnn_gd_1024_4classes/$CONFIG.py 2 \
  --work-dir ${WORKDIR} || exit

## LOG_JSON_FILE=`ls -alt ${WORKDIR}/*.log.json | cut -f10- -d" "`     #  this is not good !!!
LOG_JSON_FILE=`ls -alt ${WORKDIR}/*.log.json | head -n 1 | grep -oE '[^ ]+$'`
LOG_FILE="${LOG_JSON_FILE%.*}"
echo $LOG_JSON_FILE
echo $LOG_FILE

python ./tools/analysis_tools/analyze_logs.py plot_curve \
  --keys 0_bbox_mAP \
  --out ${WORKDIR}/log_curve.png \
   ${LOG_JSON_FILE} || exit

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
CKPT_FILE_LATEST=$WORKDIR/latest.pth
echo $WORKDIR/epoch_${epoch_num}.pth

## the following is optional
#python tools/test.py $WORKDIR/$CONFIG.py $CKPT_FILE --eval bbox \
#  --show-dir ${WORKDIR}/epoch_${epoch_num}_val || exit

# faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug2
ssh ${WIN10_IP} powershell -c mkdir ${WIN10_WORK_ROOT}/${CONFIG};
scp $LOG_FILE $LOG_JSON_FILE $LOG_CURVE_FILE $CKPT_FILE ${CKPT_FILE_LATEST} $WORKDIR/$CONFIG.py \
${WIN10_IP}:${WIN10_WORK_ROOT}/${CONFIG}

PARAMS=$(cat <<-END
set PYTHONPATH=${WIN10_GD_CODE_ROOT}\n
set SUBSET=val\n
set IMGSIZE=1024\n
set GAP=32\n
set CONFIG=${CONFIG}\n
set EPOCHNAME=epoch_${epoch_num}\n
cd ${WIN10_GD_CODE_ROOT}/mmdetection/\n
${WIN10_GD_CODE_DRIVE}\n
python demo/detect_gd1024_4classes.py ^\n
    --source E:/%SUBSET%_list.txt ^\n
    --config ${WIN10_WORK_ROOT}/%CONFIG%/%CONFIG%.py ^\n
    --weights ${WIN10_WORK_ROOT}/%CONFIG%/%EPOCHNAME%.pth ^\n
    --score-thres 0.1 ^\n
    --iou-thres 0.5 ^\n
    --hw-thres 10 ^\n
    --img-size %IMGSIZE% ^\n
    --gap %GAP% ^\n
    --save-txt ^\n
    --device 0 ^\n
    --project ${WIN10_WORK_ROOT}/%CONFIG%/outputs_%SUBSET%_%IMGSIZE%_%GAP%_%EPOCHNAME% ^\n
    --gt-xml-dir ${WIN10_GD_DATA_ROOT} ^\n
    --gt-subsize 5120 ^\n
    --gt-gap 128 ^\n
    --big-subsize 10240 ^\n
    --batchsize 4 ^\n
    --view-img\n
END
)

echo -e $PARAMS > ${WORKDIR}/run_test.bat
scp ${WORKDIR}/run_test.bat ${WIN10_IP}:${WIN10_WORK_ROOT}/${CONFIG}
echo "${WIN10_WORK_ROOT}/${CONFIG}"






