# //    Copyright 2017 Rainer Gemulla
# // 
# //    Licensed under the Apache License, Version 2.0 (the "License");
# //    you may not use this file except in compliance with the License.
# //    You may obtain a copy of the License at
# // 
# //        http://www.apache.org/licenses/LICENSE-2.0
# // 
# //    Unless required by applicable law or agreed to in writing, software
# //    distributed under the License is distributed on an "AS IS" BASIS,
# //    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# //    See the License for the specific language governing permissions and
# //    limitations under the License.
source("../../tools/mpi2.R")

## convenience method to parse and plot everything
mpi2.plot.eventlog("event.log")

## manual process (useful for filtering)
log <- mpi2.parse.eventlog("event.log")
log <- log[log$event %in% c("wait1", "wait2", "wait3"),]
events <- mpi2.process.log(log)
mpi2.plot.events(events)

## plotting to a pdf
library(rg)
rg.setDefaultOptions("pdf")
rg.options(embedFonts=F, ps=6, legend.ps=6, lwd=0.5)
rg.startplot("event.pdf")
mpi2.plot.eventlog("event.log", plot=rg.plot, legend=rg.legend)
rg.endplot()

## using a log parsed by mpi2-parse-events (much faster than in R)
## ~/mf/tools/mpi2-parse-events.sh event.log >event.tab
log <- mpi2.read.eventlog("event.tab")
events <- mpi2.process.log(log)
mpi2.plot.events(events)
