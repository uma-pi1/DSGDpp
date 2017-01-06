#!/bin/bash

NMON_BINARY=nmon

# compute name of experiment directory
DATE=`date +%y%m%d-%H%M%S`
PWD=`pwd`
BASENAME=$(basename $0)
OUTDIR="$PWD/$DATE-${BASENAME%.*}"
echo "Output directory: $OUTDIR"
mkdir $OUTDIR
export PWD
export OUTDIR

# argument 1 = nodename
start_nmon() {
    NODE=$1
    ssh -xft $NODE $NMON_BINARY -F /tmp/experiments.nmon -s 1 -c 172800 &
}
export -f start_nmon

# argument 1 = LOG_DIR, 2 = nodename
stop_nmon() {
    LOG_DIR=$1
    NODE=$2
    ssh -xft $NODE /usr/bin/killall nmon
    sleep 1s
    scp $NODE:/tmp/experiments.nmon $LOG_DIR/nmon-$NODE
    ssh -xft $NODE rm -f /tmp/experiments.nmon
    gzip -f $LOG_DIR/nmon-$NODE
}
export -f stop_nmon
