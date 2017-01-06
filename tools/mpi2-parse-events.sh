#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: Usage: mpi2-parse-events.sh <event.log> <event.tab>"
  exit
fi


grep evnt $1 | sed "s/[ ]*\([[:digit:]]\+\).*host=\([[:alnum:]]\+\) rank=\([[:digit:]]\+\) task=\([[:alnum:]]\+\) nanotime=\([[:digit:]]\+\): \([+-]\)\(.*\)/\1\t\2\t\3\t\4\t\5\t\6\t\7/" >$2

