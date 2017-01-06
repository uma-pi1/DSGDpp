//    Copyright 2017 Rainer Gemulla
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef MF_MF_H
#define MF_MF_H

#include <mpi2/mpi2.h>

#include <mf/logger.h>

#include <mf/types.h>

#include <mf/init.h>

#include <mf/lapack/lapack_wrapper.h>

#include <mf/matrix/coordinate.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/ap/aptask.h>
#include <mf/ap/apupdate.h>
#include <mf/matrix/op/copy.h>
#include <mf/matrix/op/generate.h>
#include <mf/matrix/op/sum.h>
#include <mf/matrix/op/sums.h>
#include <mf/matrix/op/sumofprod.h>
#include <mf/matrix/op/nnz.h>
#include <mf/matrix/op/crossprod.h>
#include <mf/matrix/op/unblock.h>
#include <mf/matrix/op/scale.h>
#include <mf/matrix/op/project.h>
#include <mf/matrix/op/shuffle.h>

#include <mf/matrix/io/format.h>
#include <mf/matrix/io/read.h>
#include <mf/matrix/io/write.h>
#include <mf/matrix/io/descriptor.h>
#include <mf/matrix/io/randomMatrixDescriptor.h>
#include <mf/matrix/io/load.h>
#include <mf/matrix/io/store.h>
#include <mf/matrix/io/loadDistributedMatrix.h>
#include <mf/matrix/io/mappingDescriptor.h>
#include <mf/matrix/io/ioProjected.h>
//#include <mf/matrix/io/generateDistributedMatrix.h>

#include <mf/factorization.h>
#include <mf/trace.h>

#include <mf/loss/loss.h>
#include <mf/loss/nzsl.h>
#include <mf/loss/nzrmse.h>
#include <mf/loss/l1.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzl2.h>
#include <mf/loss/sl.h>
#include <mf/loss/sl-data.h>
#include <mf/loss/sl-model.h>
#include <mf/loss/kl.h>
#include <mf/loss/gkl.h>
#include <mf/loss/gkl-data.h>
#include <mf/loss/gkl-model.h>
#include <mf/loss/biased-nzsl.h>
#include <mf/loss/biased-nzl2.h>


#include <mf/ap/lee01-gkl.h>
#include <mf/ap/dlee01-gkl.h>
#include <mf/ap/als.h>
#include <mf/ap/dals.h>
#include <mf/ap/gnmf.h>
#include <mf/ap/dgnmf.h>

#include <mf/sgd/decay/decay.h>
#include <mf/sgd/decay/decay_bolddriver.h>
#include <mf/sgd/decay/decay_sequential.h>
#include <mf/sgd/decay/decay_constant.h>

#include <mf/sgd/functions/functions.h>
#include <mf/sgd/functions/regularize-none.h>
#include <mf/sgd/functions/regularize-l1.h>
#include <mf/sgd/functions/regularize-l2.h>
#include <mf/sgd/functions/regularize-nzl2.h>
#include <mf/sgd/functions/regularize-sl.h>
#include <mf/sgd/functions/regularize-gkl.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>
#include <mf/sgd/functions/regularize-maxnorm.h>
#include <mf/sgd/functions/update-none.h>
#include <mf/sgd/functions/update-gkl.h>
#include <mf/sgd/functions/update-nzsl-nzl2.h>
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/update-sl.h>
#include <mf/sgd/functions/update-nzsl-l2.h>
#include <mf/sgd/functions/update-nzsl-l1.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>
#include <mf/sgd/functions/update-lock.h>
#include <mf/sgd/functions/update-maxnorm.h>
#include <mf/sgd/functions/update-biased-nzsl-nzl2.h>

#include <mf/sgd/sgd.h>
#include <mf/sgd/psgd.h>
#include <mf/sgd/asgd.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/dsgdpp.h>

#include <mf/sgd/decay/decay_auto.h>

#include <mf/register/register.h>
#include <mf/register/register-generated.h>

#endif
