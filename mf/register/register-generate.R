#!/usr/bin/Rscript

## class name -> includes
update <- list(UpdateNone=c("mf/sgd/functions/update-none.h"),
               UpdateNzsl=c("mf/sgd/functions/update-nzsl.h"),
               UpdateNzslL2=c("mf/sgd/functions/update-nzsl-l2.h"),
               UpdateNzslNzl2=c("mf/sgd/functions/update-nzsl-nzl2.h"),
               UpdateSl=c("mf/sgd/functions/update-sl.h"),
               UpdateGkl=c("mf/sgd/functions/update-gkl.h"),
               UpdateBiasedNzslNzl2=c("mf/sgd/functions/update-biased-nzsl-nzl2.h")
               )

## class name -> includes
regularize <- list(RegularizeNone=c("mf/sgd/functions/regularize-none.h"),
                   RegularizeL1=c("mf/sgd/functions/regularize-l1.h"),
                   RegularizeL2=c("mf/sgd/functions/regularize-l2.h"),
                   RegularizeNzl2=c("mf/sgd/functions/regularize-nzl2.h"),
                   RegularizeSl=c("mf/sgd/functions/regularize-sl.h"),
                   RegularizeGkl=c("mf/sgd/functions/regularize-gkl.h")
                   )

## class name -> includes
loss <- list(NoLoss=c("mf/loss/loss.h"),
             NzslLoss=c("mf/loss/nzsl.h"),
             NzRmseLoss=c("mf/loss/nzrmse.h"),
             GklLoss=c("mf/loss/gkl.h"),
             L1Loss=c("mf/loss/l1.h"),
             L2Loss=c("mf/loss/l2.h"),
             Nzl2Loss=c("mf/loss/nzl2.h"),
             SlLoss=c("mf/loss/sl.h"),
             BiasedNzslLoss=c(include="mf/loss/biased-nzsl.h"),
             BiasedNzl2Loss=c("mf/loss/biased-nzl2.h")
             )

## first class name -> second class name
sum.loss <- list(list("NzslLoss", c("L1Loss", "L2Loss", "Nzl2Loss")),
                 list("BiasedNzslLoss", c("BiasedNzl2Loss")) 
	);

## triples of update/regularize/loss
decay <- list(c("UpdateGkl", "RegularizeGkl", "GklLoss"),
              c("UpdateSl", "RegularizeSl", "SlLoss"),
              c("UpdateNzsl", "RegularizeNone", "NzslLoss"),
              c("UpdateNzsl", "RegularizeL1", "SumLoss_NzslLoss_L1Loss"),
              c("UpdateNzsl", "RegularizeL2", "SumLoss_NzslLoss_L2Loss"),
              c("UpdateNzsl", "RegularizeNzl2", "SumLoss_NzslLoss_Nzl2Loss"),
              c("UpdateNzslL2", "RegularizeNone", "SumLoss_NzslLoss_L2Loss"),              
              c("UpdateNzslNzl2", "RegularizeNone", "SumLoss_NzslLoss_Nzl2Loss"),             
              c("UpdateBiasedNzslNzl2", "RegularizeNone", "SumLoss_BiasedNzslLoss_BiasedNzl2Loss")
              )

group.include <- function() {
    l <- list()
    
    ## update functions
    for (i in 1:length(update)) {
        key <- paste( sort(update[[i]]), collapse="," )
        l[[key]]$include <- update[[i]]
        temp <- list()
        l[[key]]$update <- c(l[[key]]$update, update[i])
    }

    ## regularize functions
    for (i in 1:length(regularize)) {
        key <- paste( sort(regularize[[i]]), collapse="," )
        l[[key]]$include <- regularize[[i]]
        l[[key]]$regularize <- c(l[[key]]$regularize, regularize[i])
    }

    ## sum.loss functions
    all.losses <<- loss
    for (sl in sum.loss) {
        n1 <- sl[[1]]
        for (n2 in sl[[2]]) {
            inc <- sort( c(loss[[n1]], loss[[n2]]) )
            key <- paste(inc , collapse=",")
            l[[key]]$include <- inc
            temp <- list()
            name <- paste("SumLoss_", n1, "_", n2, sep="")
            temp[[name]] <- list(n1,n2)
            l[[key]]$sum.loss <- c(l[[key]]$sum.loss, temp)
            all.losses[[name]] <<- inc
        }
    }

    ## decay functions
    for (d in decay) {
        u <- d[1]
        r <- d[2]
        o <- d[3]
        inc <- c( update[u], regularize[r], all.losses[o] )
        key <- paste(inc , collapse=",")
        l[[key]]$include <- inc
        l[[key]]$decay <- c(l[[key]]$decay, list(d))
    }
    
    l
}

generate.header <- function(filename, includes=c(), def=T) {
    sink(filename)

    if (def) {
        defname <- toupper( gsub("[\\.,\\-]", "_", filename) )
    	cat("#ifndef ", defname, "\n")
        cat("#define ", defname, "\n")
    	cat("\n")
    }

    ## include headers
    cat("#include <iostream> \n")
    cat("#include <mpi2/mpi2.h> \n")
    for (i in includes) {
        cat("#include <", i, ">\n", sep="")
    }
    cat("\n")

    sink()
}

generate.footer <- function(filename, def=T) {
    sink(filename, append=T)

    if (def) {
    	cat("#endif\n")
    }

    sink() 
}


generate.update.types <- function(filename, update) {
    sink(filename, append=T)

    cat("namespace mf {\n")
    for (u in names(update)) {
        cat("typedef UpdateTruncate<", u, "> ", u, "Truncate;\n", sep="")
        cat("typedef UpdateAbs<", u, "> ", u, "Abs;\n", sep="")
        cat("typedef UpdateTruncate<UpdateAbs<", u, "> > ", u, "AbsTruncate;\n", sep="")
    }
    cat("}\n\n")

    for (u in names(update)) {
        cat("MPI2_TYPE_TRAITS(mf::", u, "Truncate)\n", sep="")
        cat("MPI2_TYPE_TRAITS(mf::", u, "Abs)\n", sep="")
        cat("MPI2_TYPE_TRAITS(mf::", u, "AbsTruncate)\n", sep="")
    }
    cat("\n")
    
    sink() 
}

generate.regularize.types <- function(filename, regularize) {
    sink(filename, append=T)

    cat("namespace mf {\n")
    for (r in names(regularize)) {
        cat("typedef RegularizeTruncate<", r, "> ", r, "Truncate;\n", sep="")
        cat("typedef RegularizeAbs<", r, "> ", r, "Abs;\n", sep="")
        cat("typedef RegularizeTruncate<RegularizeAbs<", r, "> > ", r, "AbsTruncate;\n", sep="")
    }
    cat("}\n\n")

    for (r in names(regularize)) {
        cat("MPI2_TYPE_TRAITS(mf::", r, "Truncate)\n", sep="")        
        cat("MPI2_TYPE_TRAITS(mf::", r, "Abs)\n", sep="")
        cat("MPI2_TYPE_TRAITS(mf::", r, "AbsTruncate)\n", sep="")
    }
    cat("\n")
    
    sink() 
}

generate.sum.loss <- function(filename, sum.loss) {
    sink(filename, append=T)
	templates <- c()

    cat("namespace mf {\n")
    for (l in sum.loss) {
        loss1 = l[[1]]
        for (loss2 in l[[2]]) {
        	type <- paste("SumLoss<", loss1, ", ", loss2, ">", sep="")
            cat("typedef ", type, "SumLoss_", loss1, "_", loss2, ";\n", sep="")
            templates <- c(templates, type)            
        }
    }
    cat("}\n\n")

    for (l in sum.loss) {
        loss1 = l[[1]]
        for (loss2 in l[[2]]) {
            cat("MPI2_TYPE_TRAITS2(mf::SumLoss, mf::", loss1, ", mf::", loss2, ");\n", sep="");
        }
    }
    cat("\n")

    sink() 
    return(templates)
}

generate.decay <- function(filename, decay) {
    sink(filename, append=T)

    for (d in decay) {
        u = d[[1]]
        r = d[[2]]
        l = d[[3]]
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, ", mf::", r, ", mf::", l, ");\n", sep="")
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, "Abs, mf::", r, "Abs, mf::", l, ");\n", sep="")        
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, "Truncate, mf::", r, ", mf::", l, ");\n", sep="")
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, "Truncate, mf::", r, "Truncate, mf::", l, ");\n", sep="")
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, "AbsTruncate, mf::", r, "Abs, mf::", l, ");\n", sep="")
        cat("MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::", u, "AbsTruncate, mf::", r, "AbsTruncate, mf::", l, ");\n", sep="")
    }
    cat("\n")
    
    sink()
}

generate.tasks <- function(decay) {
    tasks <- c()
    for (d in decay) {
        u = d[[1]]
        r = d[[2]]
        l = d[[3]]
        
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, ", mf::", r, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, "Abs, mf::", r, "Abs>", sep=""))
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, "Truncate, mf::", r, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, "Truncate, mf::", r, "Truncate>", sep=""))
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, "AbsTruncate, mf::", r, "Abs>", sep=""))
        tasks <- c(tasks, paste("mf::detail::DsgdTask<mf::", u, "AbsTruncate, mf::", r, "AbsTruncate>", sep=""))
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, ", mf::", r, ", mf::", l, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, "Abs, mf::", r, "Abs, mf::", l, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, "Truncate, mf::", r, ", mf::", l, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, "Truncate, mf::", r, "Truncate, mf::", l, ">", sep=""))        
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, "AbsTruncate, mf::", r, "Abs, mf::", l, ">", sep=""))
        tasks <- c(tasks, paste("mf::detail::DistributedDecayAutoTask<mf::", u, "AbsTruncate, mf::", r, "AbsTruncate, mf::", l, ">", sep=""))
    }
    
    return(tasks)
}

generate.register <- function(filename) {
	sink(filename, append=T)
	
	cat("namespace mf { namespace detail { \n")
	cat("    void registerGeneratedMatrixTasks();\n")
	cat("} }\n\n")
	
	sink()
}

generate.register.impl <- function(filename, tasks) {
    sink(filename, append=T)

    cat("namespace mf { namespace detail { \n")
    cat("void registerGeneratedMatrixTasks() {\n");
    for (task in tasks) {
        cat("    mpi2::registerTask<", task, " >();\n", sep="")
    }    
    
    cat("}\n} }\n\n")
    
    sink()
}

generate.extern <- function(filename, templates) {
	sink(filename, append=T)
	
	cat("namespace mf {\n")
	for (template in templates) {
		cat("extern template class ", template, ";\n", sep="");
	}
	cat("}\n\n")
	
	sink()
}

generate.extern.impl <- function(filenames, templates) {
	for (filename in filenames) {
		sink(filename, append=T)
		cat("namespace mf {\n")
		sink()
	}
	
	i <- 0;
	for (template in templates) {
		sink(filenames[i+1], append=T)
		cat("template class ", template, ";\n", sep="")
		sink()
		i <- (i+1) %% length(filenames)
	}
	
	for (filename in filenames) {
		sink(filename, append=T)
		cat("}\n\n")
		sink()
	}
}

groups <- group.include()

for (i in 1:length(groups)) {
    groups[[i]]$id <- i
}

## generate includes ###############################################################################

all.tasks <- c()
for (i in 1:length(groups)) {
    g <- groups[[i]]
    filename <- paste("register-generated-", i, ".h", sep="")
    includes <- unlist(g$include)
    if (!is.null(g$update)) {
        includes <- c(includes, "mf/sgd/functions/update-abs.h", "mf/sgd/functions/update-truncate.h")
    }
    if (!is.null(g$regularize)) {
        includes <- c(includes, "mf/sgd/functions/regularize-abs.h", "mf/sgd/functions/regularize-truncate.h")
    }
    if (!is.null(g$decay)) {
        for (d in g$decay) {
            includes <- c(includes, "mf/sgd/dsgd.h", "mf/sgd/decay/decay_auto.h")
            uId = groups[[ paste(update[[ d[1] ]], collapse=",") ]]$id
            rId = groups[[ paste(regularize[[ d[2] ]], collapse=",") ]]$id
            lId = groups[[ paste(all.losses[[ d[3] ]], collapse=",") ]]$id
            if (!is.null(uId) && uId != g$id) includes <- c(includes, paste("mf/register/register-generated-", uId, ".h", sep=""))
            if (!is.null(rId) && rId != g$id) includes <- c(includes, paste("mf/register/register-generated-", rId, ".h", sep=""))
            if (!is.null(lId) && lId != g$id) includes <- c(includes, paste("mf/register/register-generated-", lId, ".h", sep=""))
            includes <- c(includes, "mf/sgd/dsgd.h")
        }
    }
    include <- unique(includes)
    generate.header(filename, includes)
    generate.update.types(filename, g$update)
    generate.regularize.types(filename, g$regularize)
    templates <- generate.sum.loss(filename, g$sum.loss)
    generate.decay(filename, g$decay)
    tasks <- generate.tasks(g$decay)
    generate.extern(filename, c(templates, tasks))
    generate.footer(filename)
    groups[[i]]$templates <- templates
    groups[[i]]$tasks <- tasks
    all.tasks <- c(all.tasks, tasks)
}

filename <- "register-generated.h"
generate.header(filename, paste("mf/register/register-generated-", 1:length(groups), ".h", sep=""))
generate.register(filename)
generate.footer(filename)

## generate implementations ###########################################################################

filename <- "register-generated_impl.cc"
generate.header(filename, c("mf/register/register-generated.h"), F)
generate.register.impl(filename, all.tasks)
generate.footer(filename, F)

for (i in 1:length(groups)) {
    g <- groups[[i]]
    if ( length(c(g$templates, g$tasks)) == 0 ) next;   
    filename <- paste("register-generated-", i, "_impl.cc", sep="")
    includes <- paste("mf/register/register-generated-", i, ".h", sep="")
    generate.header(filename, includes, F)
    generate.extern.impl(filename, c(g$templates, g$tasks))
    generate.footer(filename, F)
}

cat("Generated", length(groups), " header / implementation pairs.\n")


