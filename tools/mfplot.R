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


library(ggplot2)
library(Matrix)

trace.extract <- function(trace, what) {
    if (substring(what, 1, 5) == "time$") {
        sapply(trace, function(x) x$time[[ substring(what, 6) ]])
    } else {
        sapply(trace, function(x) x[[what]])
    }
}

##'
##' Plots the progress of an matrix factorization conducted by the mf package.
##'
##' ...     : arguments of form "name=trace", where trace is a trace outputted by the mf package
mfplot <- function(...,
                   x.is=c("epoch", "it", "time$elapsed"), f.x=identity,
                   y.is=c("loss", "loss.test", "eps"), f.y=identity,
                   xlim=NULL, ylim=NULL, log="",
                   col=1:8, pch=col, lty=rep(1,20), type="b",
                   main=NA, xlab=NULL, ylab=NA,
                   myplot=plot, mylegend=legend, legend.pos="topright",
                   names=NULL, cex=1, legend.cex=1) {
    data <- list(...)
    n <- length(data)

    x <- list(); y <- list()
    if (!is.null(xlim)) range=1:xlim[2]
    for (i in 1:n) {
        x[[i]] <- f.x( trace.extract(data[[i]]$trace, x.is[1]) )
        y[[i]] <- f.y( trace.extract(data[[i]]$trace, y.is[1]) )
    }
    maxx <- max(unlist(x))
    miny <- min(unlist(y), na.rm=T)
    maxy <- max(unlist(y), na.rm=T)
    
    if (is.null(xlim)) xlim <- c(0, maxx)
    if (is.null(ylim)) ylim <- c(miny, maxy)
    if (is.null(xlab)) {
        xlab <- x.is[1]
    }

    ## remove irrelevant points
    myplot(NA, type="n", ylab=ylab,xlab=xlab, xlim=xlim, ylim=ylim, log=log, main=main, cex=cex)
    for (i in 1:n) {
        p <- which(x[[i]] >= xlim[1] & x[[i]] <= xlim[2])
        lines(x[[i]][p], y[[i]][p], type=type, pch=pch[i], col=col[i], lty=lty[i], cex=cex)
    }

    ## determine names
    if (is.null(names)) {
        names <- names(data)
        if (is.null(names)) names <- as.list(rep("", n))
        for (i in 1:n) {
            if (names[[i]]=="")
                names[[i]] <- as.character(substitute(list(...))[[1+i]])
        }
    }

    ## plot legend
    if (!is.null(mylegend)) {
        if (type == "b") {
            mylegend(legend.pos, legend=names, pch=pch, col=col, bg="white", cex=legend.cex)
        } else {
            mylegend(legend.pos, legend=names, lty=1, col=col, bg="white", cex=legend.cex)
        }
    }
    
    invisible(y)
}

mfheatmap <- function(A, Rowv=NA, Colv=NA, scale="none", col=cm.colors(256), ...) {
    heatmap(A, Rowv=Rowv, Colv=Colv, scale=scale, col=col, ...)
}

read.mma <- function(filename) {
    ## TODO: error handling
    f <- file(filename, "r")

    ## read size
    v <- scan(f, what=integer(0), n=2, comment.char="%", quiet=T);
    m <- v[1]
    n <- v[2]

    v <- array(scan(f, what=double(0), n=n*m, comment.char="%", quiet=T), c(m,n))

    close(f)
    v
}

read.mmc <- function(filename) {
    readMM(filename)
}

write.mmc <- function(M, filename) {
    writeMM(M, filename)
}

## reads a csv file with (row, column, value) triplets into a data frame with
## columns i, j, x
read.csv.frame <- function(file) {
    x <- read.csv(file)
    data.frame(i=as.integer(x[,1]), j=as.integer(x[,2]), x=as.double(x[,3]))
}

## takes a data frame with columns i,j,x and writes an mmc file
write.mmc.frame <- function(M, file, m=max(M$i), n=max(M$j)) {
    sink(file)
    cat("%%MatrixMarket matrix coordinate real general\n")
    cat("%=================================================================================\n")
    cat("%\n")
    cat("% This ASCII file represents a sparse MxN matrix with L \n")
    cat("% nonzeros in the following Matrix Market format:\n")
    cat("%\n")
    cat("% +----------------------------------------------+\n")
    cat("% |%%MatrixMarket matrix coordinate real general | <--- header line\n")
    cat("% |%                                             | <--+\n")
    cat("% |% comments                                    |    |-- 0 or more comment lines\n")
    cat("% |%                                             | <--+         \n")
    cat("% |    M  N  L                                   | <--- rows, columns, entries\n")
    cat("% |    I1  J1  A(I1, J1)                         | <--+\n")
    cat("% |    I2  J2  A(I2, J2)                         |    |\n")
    cat("% |    I3  J3  A(I3, J3)                         |    |-- L lines\n")
    cat("% |        . . .                                 |    |\n")
    cat("% |    IL JL  A(IL, JL)                          | <--+\n")
    cat("% +----------------------------------------------+   \n")
    cat("%\n")
    cat("% Indices are 1-based, i.e. A(1,1) is the first element.\n")
    cat("%\n")
    cat("%=================================================================================\n")
    cat(m, n, nrow(M), "\n")
    write.table(M, col.names=F, row.names=F, sep=" ")
    sink()
}

write.mma <- function(M, file) {
    sink(file)
    cat("%%MatrixMarket matrix array real general\n")
    cat("% First line: ROWS COLUMNS\n")
    cat("% Subsequent lines: entries in column-major order\n")
    cat(nrow(M), ncol(M), "\n")
    write.table(as.vector(M), col.names=F, row.names=F)
    sink()
}

read.index <- function(filename) {
    scan(filename, what=list(key=integer(0), value=""), sep="\t", multi.line=F, quote="")
}

read.index.pair <- function(filename) {
    scan(filename, what=list(key=integer(0), value1="", value2=""), sep="\t", multi.line=F, quote="")
}

read.map <- function(filename, sep=" ") {
    scan(filename, what=list(key=integer(0), value=integer(0)), sep=sep, multi.line=F, quote="", skip=1)
}

cos.dist <- function(A, dist=T) {
    m <- nrow(A)
    D <- array(0, c(m,m))

    for (i in 1:m) {
        for (j in 1:m) {
            D[i,j] <- 1-crossprod(A[i,], A[j,]) / sqrt(crossprod(A[i,]))/ sqrt(crossprod(A[j,]))
        }
    }

    if (dist) { as.dist(D) } else { D }
}

cos.dist.exp <- function(A) cos.dist(exp(A))

## returns permutations of the columns of H by closeness to p
similar.cols <- function(j, H) {
    p <- H[,j]
    scp <- sqrt(crossprod( p ))
    similarity <- apply(H, 2, function(x) {
        sc <- sqrt(crossprod(x))
        ifelse(sc==0, 0, crossprod(p, x) / scp / sc)
    })
    o <- order(similarity, decreasing=T)
    s <- similarity[o]
    data.frame(index=o, name=names(s), similarity=s, row.names=NULL)
}

## returns permutations of the columns of H by closeness to p
row.predictions <- function(j, W, H) {
    p <- H[,j]
    dim(p) <- c(length(p),1)
    similarity <- W %*% p
    o <- order(similarity, decreasing=T)
    s <- similarity[o]
    data.frame(index=o, name=rownames(W)[o], prediction=s, row.names=NULL)
}
