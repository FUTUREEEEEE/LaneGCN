num_process=$1
g=$(($2<8?$2:8))
START_TIME=`date +%Y%m%d-%H:%M:%S`
echo '[USAGE] sh train.sh <CONFIG> <GPU_NUM> <JOB_NAME>'
mkdir -p logs
LOG_FILE=logs/train-log-$START_TIME

srun  -p caif_debug \
      -n $num_process\
      --gres=gpu:$g\
      --ntasks-per-node=$g \
      --cpus-per-task=16 \
      --job-name=complete_run \
python  mytrain_ddp.py -m lanegcn  \
  2>&1 | tee $LOG_FILE 1
echo -e "\033[32m[ Please see LOG_FILE for details: \"tail -f ${LOG_FILE}\" ]\033[0m"
