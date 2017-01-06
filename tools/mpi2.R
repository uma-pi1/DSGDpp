## parses a log file containing events
mpi2.parse.eventlog <- function(filename) {
    con <- file(filename) 
    open(con);
    lines <- readLines(con, warn = FALSE)
    close(con)

    data <- NULL
    indexes <- grep("evnt", lines)
    matches <- gsub("[[:space:]]*([[:alnum:]]+)[[:space:]]+evnt[[:space:]]+\\|[[:space:]]+host=([[:alnum:]]+) rank=([[:digit:]]+) task=([[:alnum:]]+) nanotime=([[:digit:]]+): ([+-])(.+)",
                    "\\1\t\\2\t\\3\t\\4\t\\5\t\\6\t\\7",
                    lines[indexes])
    for (match in matches) {
        record <- strsplit(match, "\t")[[1]]
        if (length(record) == 7) {
            data <- rbind(data, record)
        }
    }
    
    result <- data.frame(elapsedTime=as.integer(data[,1]),
                         host=data[,2],
                         rank=as.integer(data[,3]),
                         task=data[,4],
                         time=as.double(data[,5]) / 1E6,
                         type=data[,6],
                         event=data[,7])
    result
}

## reads a log that has been parsed outside using mpi2-parse-log.sh
mpi2.read.eventlog <- function(filename) {
    result <- read.table(filename, col.names=c("elapsedTime", "host", "rank", "task", "time", "type", "event"))
    cat(nrow(result), "events\n")
    result$time=as.double(result$time) / 1E6
    result
}

mpi2.sections <- function(events) {
    rank <- -1
    task <- -1
    running.sections <- list()
    events <- events[order(events$rank, events$task, events$time),]
    sections <- NULL
    for (i in 1:nrow(events)) {
        if (i %% 1000 == 0) cat("Processed", i, "of", nrow(events), "events\n")
        event <- events[i,]
        
        ## new rank or task
        if (events[i,3] != rank || events[i,4] != task) {
            if (length(running.sections) != 0) {
                stop("Invalid input")
            }
            rank <- events[i,3]
            task <- events[i,4]       
        }
        
        ## processes start events
        if (event$type == "+") {
            running.sections <- c(running.sections, list(list(event=event$event, begin=event$time)))
        }
        
        ## process end events
        if (event$type == "-") {
            if (length(running.sections) == 0) stop("Invalid input: no open event")
            section <- tail(running.sections, 1)[[1]]
            if (section$event != event$event) stop("Invalid input: event ", event$event, " closes section ", section$event, " in row ", i)
            sections <- rbind(sections, data.frame(rank=rank, task=task, event=section$event,
                                                   begin=section$begin, end=event$time,
                                                   depth=length(running.sections)))
            running.sections[[length(running.sections)]] <- NULL
        }
    }

    ## print some information
    counts <- aggregate(sections$event, by=list(event=sections$event), length)
    names(counts)[2] <- "frequency"
    tmin <- aggregate(sections$end-sections$begin, by=list(event=sections$event), min)
    names(tmin)[2] <- "tmin"
    tmean <- aggregate(sections$end-sections$begin, by=list(event=sections$event), mean)
    names(tmean)[2] <- "tmean"
    tmax <- aggregate(sections$end-sections$begin, by=list(event=sections$event), max)
    names(tmax)[2] <- "tmax"
    tcv <- aggregate(sections$end-sections$begin, by=list(event=sections$event), sd)
    names(tcv)[2] <- "tcv" ## coefficient of variation
    frame <- merge( merge( merge(counts, tmin), merge(tmean, tmax) ), tcv)
    frame$tcv = frame$tcv/frame$tmean
    print(frame)
    
    sections
}

## extracts information about tasks from an event log
mpi2.tasks <- function(events) {
    ## construct a rank / task / begin / end data frame
    tasks <- list(begin=aggregate(events$time, by=list(rank=events$rank, task=events$task), min),
                  end=aggregate(events$time, by=list(rank=events$rank, task=events$task), max))
    tasks <- data.frame(rank=tasks$begin$rank, task=tasks$begin$task, begin=tasks$begin$x, end=tasks$end$x)
    tasks <- tasks[order(tasks$rank, tasks$begin, tasks$task), ]

    ## assign each task a slot (such that no two tasks run at the same time in a slot)
    slots <- c()
    rank <- -1
    for (i in 1:nrow(tasks)) {
        task <- tasks[i,]

        ## a new rank starts
        if (task$rank != rank) {
            rank <- task$rank
            running.tasks <- c()
        }

        ## get a slot
        slot <- -1
        if (length(running.tasks)>0) {
            for (i in 1:length(running.tasks)) {
                end <- running.tasks[i]
                if (is.na(end) || end < task$begin) {
                    slot <- i
                    running.tasks[slot] <- task$end
                    break
                }
            }
        }
        if (slot == -1) {
            running.tasks <- c(running.tasks, task$end)
            slot <- length(running.tasks)
        }
        slots <- c(slots, slot)
    }
    tasks$slot <- slots
    tasks
}




mpi2.process.log = function(events) {
    tasks <- mpi2.tasks(events)
    sections <- mpi2.sections(events)

    ## determine the maximum depth per task
    depths <- aggregate(sections$depth, by=list(rank=sections$rank, task=sections$task), max)
    names(depths)[3] <- "depth"
    tasks <- merge(tasks, depths)
        
    ## determine the maximum depth per slot
    depths <- aggregate(tasks$depth, by=list(rank=tasks$rank, slot=tasks$slot), max)
    names(depths)[3] <- "slotdepth"
    tasks <- merge(tasks, depths)

    ## determine the offset of each slot
    offsets <- unique( tasks[, c("rank", "slot", "slotdepth")] )
    offsets <- offsets[ order(offsets$rank, offsets$slot), ]
    offsets$slotoffset <- cumsum(c(0, head(offsets$slotdepth, -1))) + offsets$rank
    tasks <- merge(tasks, offsets)

    sections <- merge(sections, tasks[, c("rank", "task", "slot", "slotoffset")])

    list(log=events, tasks=tasks, sections=sections)    
}

mpi2.plot.events <- function(processed.events,
                             begin = min(processed.events$tasks$begin),
                             end = max(processed.events$tasks$end),
                             plot = graphics::plot,
                             legend = graphics::legend)
{    
    ## adjust the times and convert to seconds
    tasks <- processed.events$tasks
    tasks$begin <- (tasks$begin - begin)/ 1000.
    tasks$end <- (tasks$end - begin)/ 1000.
    sections <- processed.events$sections
    sections$begin <- (sections$begin - begin)/ 1000.
    sections$end <- (sections$end - begin)/ 1000.
    begin <- begin/1000
    end <- end/1000
    
    ## graph limits
    xlim=c(0,end-begin)
    ylim=c(0, max(tasks$slotoffset*1.25 + tasks$depth)*1.07)
    
    plot(NA, xlim=xlim, ylim=ylim, xlab="Time (s)", ylab="")
    
    ## plot the tasks
    for (i in 1:nrow(tasks)) {
        task <- tasks[i,]
        rect(task$begin, task$slotoffset*1.25, task$end, task$slotoffset*1.25+task$depth)
    }

    ## plot the sections
    for (i in 1:nrow(sections)) {
        section <- sections[i,]
        rect(section$begin, section$slotoffset*1.25+section$depth-.95,
             section$end, section$slotoffset*1.25+section$depth-.05,
             col=section$event)
    }

    ## plot lines between ranks
    offsets <- aggregate(tasks$slotoffset, by=list(rank=tasks$rank), min)
    for (offset in offsets$x) {
        lines(xlim, rep(offset*1.25-0.75, 2), lty=2)
    }
    
    ## legend
    legend("topleft", as.character(levels(section$event)), fill=1:length(levels(section$event)), ncol=5)
}

mpi2.plot.eventlog <- function(filename, ...) {
    if (length( grep(".log", filename) )>0) {
        log <- mpi2.parse.eventlog(filename)
    } else {
        log <- mpi2.read.eventlog(filename)
    }
    events <- mpi2.process.log(log)
    mpi2.plot.events(events, ...)
}



