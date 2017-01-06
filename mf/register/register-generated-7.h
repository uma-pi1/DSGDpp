#ifndef  REGISTER_GENERATED_7_H 
#define  REGISTER_GENERATED_7_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-biased-nzsl-nzl2.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateBiasedNzslNzl2> UpdateBiasedNzslNzl2Truncate;
typedef UpdateAbs<UpdateBiasedNzslNzl2> UpdateBiasedNzslNzl2Abs;
typedef UpdateTruncate<UpdateAbs<UpdateBiasedNzslNzl2> > UpdateBiasedNzslNzl2AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateBiasedNzslNzl2Truncate)
MPI2_TYPE_TRAITS(mf::UpdateBiasedNzslNzl2Abs)
MPI2_TYPE_TRAITS(mf::UpdateBiasedNzslNzl2AbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
