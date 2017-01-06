#ifndef  REGISTER_GENERATED_2_H 
#define  REGISTER_GENERATED_2_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateNzsl> UpdateNzslTruncate;
typedef UpdateAbs<UpdateNzsl> UpdateNzslAbs;
typedef UpdateTruncate<UpdateAbs<UpdateNzsl> > UpdateNzslAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateNzslTruncate)
MPI2_TYPE_TRAITS(mf::UpdateNzslAbs)
MPI2_TYPE_TRAITS(mf::UpdateNzslAbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
