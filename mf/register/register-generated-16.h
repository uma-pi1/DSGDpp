#ifndef  REGISTER_GENERATED_16_H 
#define  REGISTER_GENERATED_16_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/loss/nzl2.h>
#include <mf/loss/nzsl.h>

namespace mf {
}


namespace mf {
}


namespace mf {
typedef SumLoss<NzslLoss, Nzl2Loss>SumLoss_NzslLoss_Nzl2Loss;
}

MPI2_TYPE_TRAITS2(mf::SumLoss, mf::NzslLoss, mf::Nzl2Loss);


namespace mf {
extern template class SumLoss<NzslLoss, Nzl2Loss>;
}

#endif
