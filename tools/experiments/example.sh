#!/bin/sh

IFS=","
HOSTNAMES=`hostname`
HOSTS=`echo ${HOSTNAMES//","/" "} | wc -w`
TASKSPERRANK=2

. include.sh

cp $0 $OUTDIR/

for HOSTNAME in $HOSTNAMES; do
    start_nmon $HOSTNAME
done

mpirun --hosts $HOSTNAMES ~/svnrepos/devel/c++/mf/build/tools/mfdsgd --tasksPerRank $TASKSPERRANK --epochs 5 --trace $OUTDIR/dsgd-${HOSTS}x$TASKSPERRANK.R --traceVar dsgd ~/data/netflix/probe.perm.mmc | tee $OUTDIR/stdout.log

for HOSTNAME in $HOSTNAMES; do
    stop_nmon $OUTDIR/ $HOSTNAME
done
