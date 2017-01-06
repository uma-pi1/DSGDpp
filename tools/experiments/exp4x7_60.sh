#!/bin/sh

IFS=","
HOSTNAMES="d5blade22,d5blade23,d5blade24,d5blade25"
HOSTS=`echo ${HOSTNAMES//","/" "} | wc -w`
TASKSPERRANK=7
MFDSGD_PATH=/home/chteflio/workspace/mf/build/examples/fmakari
MATRIX_PATH=/home/chteflio/InitialMatrices_sorted_averagedOut
FACTOR_PATH=/home/chteflio/InitialFactors_Rank60

. include.sh

cp $0 $OUTDIR/

for HOSTNAME in $HOSTNAMES; do
    start_nmon $HOSTNAME
done

sleep 2s
mpirun \
    --hosts "$HOSTNAMES" $MFDSGD_PATH/mymfdsgd \
    --input-file $MATRIX_PATH/train.perm.sorted.av.mmc --input-test-file $MATRIX_PATH/validate.perm.sorted.av.mmc \
    --input-row-file $FACTOR_PATH/W.mma --input-col-file $FACTOR_PATH/H.mma \
    --tasksPerRank $TASKSPERRANK \
    --epochs 1 \
    --rank 60 \
    --trace $OUTDIR/dsgd-${HOSTS}x$TASKSPERRANK.R --traceVar dsgd-${HOSTS}x$TASKSPERRANK \
    --update "NZSL(-1000,1000)" --regularize "NZL2(0.01)" --loss "NZL2(0.01)" \
    --decay "Auto(0.01,28,/home/chteflio/InitialMatrices_sorted_averagedOut/trainSample.mfp)" \
| tee $OUTDIR/stdout.log

for HOSTNAME in $HOSTNAMES; do
    stop_nmon $OUTDIR/ $HOSTNAME
done
