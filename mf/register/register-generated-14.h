#ifndef  REGISTER_GENERATED_14_H 
#define  REGISTER_GENERATED_14_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/loss/l1.h>
#include <mf/loss/nzsl.h>

namespace mf {
}


namespace mf {
}


namespace mf {
typedef SumLoss<NzslLoss, L1Loss>SumLoss_NzslLoss_L1Loss;
}

MPI2_TYPE_TRAITS2(mf::SumLoss, mf::NzslLoss, mf::L1Loss);


namespace mf {
extern template class SumLoss<NzslLoss, L1Loss>;
}

#endif
