#ifndef  REGISTER_GENERATED_17_H 
#define  REGISTER_GENERATED_17_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/loss/biased-nzl2.h>
#include <mf/loss/biased-nzsl.h>

namespace mf {
}


namespace mf {
}


namespace mf {
typedef SumLoss<BiasedNzslLoss, BiasedNzl2Loss>SumLoss_BiasedNzslLoss_BiasedNzl2Loss;
}

MPI2_TYPE_TRAITS2(mf::SumLoss, mf::BiasedNzslLoss, mf::BiasedNzl2Loss);


namespace mf {
extern template class SumLoss<BiasedNzslLoss, BiasedNzl2Loss>;
}

#endif
