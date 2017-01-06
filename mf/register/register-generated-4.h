#ifndef  REGISTER_GENERATED_4_H 
#define  REGISTER_GENERATED_4_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl-nzl2.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateNzslNzl2> UpdateNzslNzl2Truncate;
typedef UpdateAbs<UpdateNzslNzl2> UpdateNzslNzl2Abs;
typedef UpdateTruncate<UpdateAbs<UpdateNzslNzl2> > UpdateNzslNzl2AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateNzslNzl2Truncate)
MPI2_TYPE_TRAITS(mf::UpdateNzslNzl2Abs)
MPI2_TYPE_TRAITS(mf::UpdateNzslNzl2AbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
