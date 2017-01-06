#!/bin/sh
#
# Plots the events in the specified log file

MPI2_TOOLS_HOME="`dirname $0`"

if [ $# -ne 1 ]; then
  echo "Usage: Usage: mpi2-plot-events.sh <event.log>"
  exit
fi

LOG_FILE=$1
BASE_NAME="${LOG_FILE%.*}"
TAB_FILE=$BASE_NAME.tab
PDF_FILE=$BASE_NAME.pdf

$MPI2_TOOLS_HOME/mpi2-parse-events.sh $LOG_FILE $TAB_FILE

echo "
library(rg)
source(\"$MPI2_TOOLS_HOME/mpi2.R\")
rg.setDefaultOptions(\"pdf\")
rg.options(embedFonts=F, ps=6, legend.ps=6, lwd=0.5)
rg.startplot(\"$PDF_FILE\")
mpi2.plot.eventlog(\"$TAB_FILE\", plot=rg.plot, legend=rg.legend)
rg.endplot()
" | Rscript -

