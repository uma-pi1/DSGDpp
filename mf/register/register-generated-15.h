#ifndef  REGISTER_GENERATED_15_H 
#define  REGISTER_GENERATED_15_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/loss/l2.h>
#include <mf/loss/nzsl.h>

namespace mf {
}


namespace mf {
}


namespace mf {
typedef SumLoss<NzslLoss, L2Loss>SumLoss_NzslLoss_L2Loss;
}

MPI2_TYPE_TRAITS2(mf::SumLoss, mf::NzslLoss, mf::L2Loss);


namespace mf {
extern template class SumLoss<NzslLoss, L2Loss>;
}

#endif
