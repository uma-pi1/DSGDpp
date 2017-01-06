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
#!/usr/bin/Rscript

## list of update functions
## each element is a (name, implementation) pair with the following fields:
##   class: name of class implementing the update
##   args : (minimum, maximum) number of double-valued arguments
update <- list(
               None=list(class="UpdateNone", args=c(0,0)),
               Nzsl=list(class="UpdateNzsl", args=c(0,0)),
               "Nzsl_L2" = list(class="UpdateNzslL2", args=c(1,1)),
               "Nzsl_Nzl2" = list(class="UpdateNzslNzl2", args=c(1,1)),
               "Biased_Nzsl_Nzl2" = list(class="UpdateBiasedNzslNzl2", args=c(4,4)),
               SlData = list(class="UpdateSl", args=c(0,0)),
               GklData=list(class="UpdateGkl", args=c(0,0))
               )

## list of regularize functions
## same format as above
regularize <- list(
                   None=list(class="RegularizeNone", args=c(0,0)),
#                    L1=list(class="RegularizeL1", args=c(1,1)),
#                    L2=list(class="RegularizeL2", args=c(1,1)),
                   Nzl2=list(class="RegularizeNzl2", args=c(1,1)),
                   SlModel=list(class="RegularizeSl", args=c(0,0)),
                   GklModel=list(class="RegularizeGkl", args=c(0,0))
                   )

## list of loss functions
## same format as above
loss <- list(
             None=list(class="NoLoss", args=c(0,0)),
             Nzsl=list(class="NzslLoss", args=c(0,0)),
             NzRMSE=list(class="NzRmseLoss", args=c(0,0)),
#              L1=list(class="L1Loss", args=c(1,1)),
             L2=list(class="L2Loss", args=c(1,1)),
             Nzl2=list(class="Nzl2Loss", args=c(1,1)),
             BiasedNzsl=list(class="BiasedNzslLoss", args=c(4,4)),
             BiasedNzl2=list(class="BiasedNzl2Loss", args=c(4,4)),
             Sl=list(class="SlLoss", args=c(0,0)),             
             Gkl=list(class="GklLoss", args=c(0,0))
             )


## loss functions defined as sums 
sum.loss <- list(
#                  "Nzsl_L1"=list(class1="NzslLoss", class2="L1Loss", args=c(1,1)),
                 "Nzsl_L2"=list(class1="NzslLoss", class2="L2Loss", args=c(1,1)),
                 "Nzsl_Nzl2"=list(class1="NzslLoss", class2="Nzl2Loss", args=c(1,1)),
                 "Biased_Nzsl_Nzl2"=list(class1="BiasedNzslLoss", class2="BiasedNzl2Loss", args=c(4,4))
                 );

## combinations to instantiate
combinations <- list(
                     list(update="GklData", regularize="GklModel", loss="Gkl"),
                     list(update="SlData", regularize="SlModel", loss="Sl"),
                     list(update="Nzsl", regularize="None", loss="Nzsl"),
                     #list(update="Nzsl", regularize="L1", loss="Nzsl_L1"),
                     #list(update="Nzsl", regularize="L2", loss="Nzsl_L2"),
                     list(update="Nzsl_L2", regularize="None", loss="Nzsl_L2"),
                     #list(update="Nzsl", regularize="Nzl2", loss="Nzsl_Nzl2"),
                     list(update="Nzsl_Nzl2", regularize="None", loss="Nzsl_Nzl2"),
                     list(update="Biased_Nzsl_Nzl2", regularize="None", loss="Biased_Nzsl_Nzl2")
                     )
                                                   
#%--------------------------------------------
generate.header <- function(filename, include=T) {
    sink(filename)

    if (include) {
    	cat("#ifndef MFDSGD_GENERATED_H \n")
    	cat("#define MFDSGD_GENERATED_H \n")
    	cat("\n")
    }

    ## include headers
    cat("#include <iostream> \n")
    cat("#include <mpi2/mpi2.h> \n")
    cat("#include <mf/mf.h> \n")
    cat("#include \"mfdsgd-args.h\" \n")
    if (!include) {
        cat("#include <cmath> \n")
    	cat("#include \"mfdsgd-generated.h\" \n")
        cat("#include \"mfdsgd-run.h\" \n")
    }
    cat("\n")

    sink()
}

generate.footer <- function(filename, include=T) {
    sink(filename, append=T)

    cat("\n")
    if (include) {
    	cat("#endif\n")
    }

    sink() 
}

generate.update.types <- function(filename) {
    sink(filename, append=T)

    cat("namespace mf {\n")
    for (u in update) {
        cat("typedef UpdateTruncate<", u$class, "> ", u$class, "Truncate;\n", sep="")
#         cat("typedef UpdateAbs<", u$class, "> ", u$class, "Abs;\n", sep="")
#         cat("typedef UpdateTruncate<UpdateAbs<", u$class, "> > ", u$class, "AbsTruncate;\n", sep="")
    }
    cat("}\n\n")
    
    sink() 
}

generate.regularize.types <- function(filename) {
    sink(filename, append=T)

    cat("namespace mf {\n")
    for (r in regularize) {
        cat("typedef RegularizeTruncate<", r$class, "> ", r$class, "Truncate;\n", sep="")
#         cat("typedef RegularizeAbs<", r$class, "> ", r$class, "Abs;\n", sep="")
#         cat("typedef RegularizeTruncate<RegularizeAbs<", r$class, "> > ", r$class, "AbsTruncate;\n", sep="")
    }
    cat("}\n\n")
    
    sink() 
}

generate.sum.loss <- function(filename) {
    sink(filename, append=T)
    templates <- c()

    cat("namespace mf {\n")
    for (l in sum.loss) {
        type <- paste("SumLoss<", l$class1, ", ", l$class2, ">", sep="")
        cat("typedef ", type, " SumLoss_", l$class1, "_", l$class2, ";\n", sep="")
        templates <- c(templates, type)            
    }
    cat("}\n\n")

    sink() 
    return(templates)
}

generate.combinations <- function() {
    templates <- c()
    for (c in combinations) {
        ## determine update, regularize, loss
        u <- update[[c$update]]$class
        r <- regularize[[c$regularize]]$class
        if (c$loss %in% names(loss)) {
            l = loss[[c$loss]]$class
        } else {
            l = sum.loss[[c$loss]]
            l <- paste("SumLoss<", l$class1, ", ", l$class2, ">", sep="")
        }

        ## create distributed SGD job
        templates <- c(templates, paste("mf::DsgdJob<mf::", u, ", mf::", r, ">", sep=""))
#         templates <- c(templates, paste("mf::DsgdJob<mf::", u, "Abs, mf::", r, "Abs>", sep=""))
        templates <- c(templates, paste("mf::DsgdJob<mf::", u, "Truncate, mf::", r, ">", sep=""))
        templates <- c(templates, paste("mf::DsgdJob<mf::", u, "Truncate, mf::", r, "Truncate>", sep=""))
#         templates <- c(templates, paste("mf::DsgdJob<mf::", u, "AbsTruncate, mf::", r, "Abs>", sep=""))
#         templates <- c(templates, paste("mf::DsgdJob<mf::", u, "AbsTruncate, mf::", r, "AbsTruncate>", sep=""))

        ## create decay functions
        templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, ", mf::", r, ", mf::", l, " >", sep=""))
#         templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, "Abs, mf::", r, "Abs, mf::", l, " >", sep=""))
        templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, "Truncate, mf::", r, ", mf::", l, " >", sep=""))
        templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, "Truncate, mf::", r, "Truncate, mf::", l, " >", sep=""))
#         templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, "AbsTruncate, mf::", r, "Abs, mf::", l, " >", sep=""))
#         templates <- c(templates, paste("mf::DistributedDecayAuto< mf::", u, "AbsTruncate, mf::", r, "AbsTruncate, mf::", l, " >", sep=""))
    }

    return(templates)
}

generate.runArgs <- function(filename) {
    sink(filename, append=T)
    cat("bool runArgs(Args& args);")
    cat("\n")
    sink()
}

cond.abs <- "(args.abs)"
cond.truncate <- "(!std::isnan(args.truncateArgs[0]) && !std::isnan(args.truncateArgs[1]))"

generate.instance <- function(type, name, u, abs, truncate) {
    cap <- function(s) paste(toupper(substr(s, 1, 1)), substr(s, 2, nchar(s)), sep="")
    
    cat("\t\t\t\tif (args.", type, "Args.size()<", u$args[1], " || args.", type, "Args.size()>", u$args[2], ") {\n", sep="")
    cat("\t\t\t\t\tstd::cout << \"Invalid number of arguments in \" << args.", type, "String << std::endl;\n", sep="")
    cat("\t\t\t\t\treturn false;\n")
    cat("\t\t\t\t}\n")
    cat("\t\t\t\t", u$class, " ", type, "(UNINITIALIZED);\n", sep="")
    for (count in u$args[1]:u$args[2]) {
        cat("\t\t\t\tif (args.", type, "Args.size()==", count, ") {\n", sep="")
        cat("\t\t\t\t\t", type, " = ", u$class, "(", sep="")
        sep=""
        if (count > 0) {
            for (i in 1:count) {
         		if (count > 1) {
					if (i == count) {
						sep=""
					} else {
						sep=", "
					}
				}
                cat("args.", type, "Args[", i-1, "]", sep, sep="")
                #sep=", "
            }
        }
        cat(");\n");
        cat("\t\t\t\t}\n")
    }

    typeAbs <- ""
    if (abs) {
        typeAbs = paste(cap(type), "Abs<", u$class, ">", sep="")
        cat("\t\t\t\t", typeAbs, " ", type, "Abs(", type, ");\n", sep="")                    
    } else {
        typeAbs = u$class
        cat("\t\t\t\t", typeAbs, " ", type, "Abs = ", type, ";\n", sep="")
    }
    typeTruncate <- ""
    if (truncate) {
        typeTruncate = paste(cap(type), "Truncate<", typeAbs, " >", sep="")
        cat("\t\t\t\t", typeTruncate, " ", type, "Truncate(", type, "Abs, args.truncateArgs[0], args.truncateArgs[1]);\n", sep="")                    
    } else {
        typeTruncate = typeAbs
        cat("\t\t\t\t", typeAbs, " ", type, "Truncate = ", type, "Abs;\n", sep="")
    }
    ## cat("\t\t\t\ttypedef ", typeTruncate, " ", cap(type), ";\n", sep="")
}

generate.loss <- function(name, u, var) {
    type="loss"
    cat("\t\t\t\tif (args.",type,"Args.size()<", u$args[1], " || args.", type, "Args.size()>", u$args[2], ") {\n", sep="")
    cat("\t\t\t\t\tstd::cout << \"Invalid number of arguments in \" << args.", type, "String << std::endl;\n", sep="")
    cat("\t\t\t\t\treturn false;\n")
    cat("\t\t\t\t}\n")
    cat("\t\t\t\t", u$class, " ", var, "(UNINITIALIZED);\n", sep="")
    for (count in u$args[1]:u$args[2]) {
        cat("\t\t\t\tif (args.", type, "Args.size()==", count, ") {\n", sep="")
        cat("\t\t\t\t\t", var, " = ", u$class, "(", sep="")
        sep=""
        if (count > 0) {
            for (i in 1:count) {
            	if (count > 1) {
					if (i == count) {
						sep=""
					} else {
						sep=", "
					}
				}
                cat("args.", type, "Args[", i-1, "]", sep, sep="")
                #sep=", "
            }
        }
        cat(");\n");
        cat("\t\t\t\t}\n")
    }

}

generate.runArgs.impl <- function(filename) {
    sink(filename, append=T)

    cat("\n")
    cat("bool runArgs(Args& args) {\n", sep="")
    cat("\tusing namespace mf;\n")
    cat("\tusing namespace mpi2;\n")
    for (abs in c(F,T)) {
        cat("\tif (", ifelse(abs, "", "!"), cond.abs, ") {\n", sep="")
        for (truncate in c(F,T)) {
            cat("\t\tif (", ifelse(truncate, "", "!"), cond.truncate, ") {\n", sep="")

            for (c in combinations) {
                ## determine update, regularize, loss
                u <- update[[c$update]]
                r <- regularize[[c$regularize]]
                if (c$loss %in% names(loss)) {
                    l = loss[[c$loss]]
                    l.class = l$class
                } else {
                    l = sum.loss[[c$loss]]
                    l.class = paste("SumLoss<", l$class1, ", ", l$class2, ">", sep="")
                }

                ## generate condition
                cat("\t\t\tif (")
                cat("(args.updateName.compare(\"", c$update, "\") == 0)", sep="")
                cat(" && ")
                cat("(args.regularizeName.compare(\"", c$regularize, "\") == 0)", sep="")
                cat(" && ")
                cat("(args.lossName.compare(\"", c$loss, "\") == 0)", sep="")
                cat(") {\n")

                ## generate update
                generate.instance("update", c$update, u, abs, truncate)
                generate.instance("regularize", c$regularize, r, abs, truncate)

                ## generate loss
                if (c$loss %in% names(loss)) {
                    generate.loss(c$loss, l, "loss")
                } else {
                    cat("\t\t\t\t", l$class1, " loss1;\n", sep="")
                    generate.loss(c$loss, list(class=l$class2, args=l$args), "loss2")
                    cat("\t\t\t\t", l.class, " loss = ", l.class, "(loss1, loss2);\n", sep="")
                }

                ## run
                cat("\t\t\t\trunDsgd(args, updateTruncate, regularizeTruncate, loss);\n")
                cat("\t\t\t\treturn true;\n")
                ## end condition
                cat("\t\t\t}\n")
            }
        cat("\t\t}\n")
        }
        cat("\t}\n")
    }

    cat("\tstd::cerr << \"Invalid combination of update, regularize, and loss arguments\" << std::endl;\n")
    cat("\tstd::cout << \"Valid combinations are:\" << std::endl;\n")
    for (c in combinations) {
        cat("\tstd::cout << \"\t", c$update, " / ", c$regularize, " / ", c$loss, "\" << std::endl;\n", sep="")
    }
    cat("\treturn false;\n");
    cat("}\n")
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

#% for mfdsgd_generate.h -----------------------------------
filename <- "mfdsgd-generated.h"
generate.header(filename)
generate.update.types(filename)
generate.regularize.types(filename)
templates <- generate.sum.loss(filename)
templates <- c(templates, generate.combinations())
generate.extern(filename, templates)

generate.runArgs(filename)
generate.footer(filename)

filename <- "mfdsgd-generated_impl.cc"
generate.header(filename, F)
generate.runArgs.impl(filename)
generate.footer(filename, F)

filenames <- paste("mfdsgd-generated_impl", 1:8, ".cc", sep="")
for (filename in filenames) generate.header(filename, F)
generate.extern.impl(filenames, templates)
for (filename in filenames) generate.footer(filename, F)




