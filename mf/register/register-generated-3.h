#ifndef  REGISTER_GENERATED_3_H 
#define  REGISTER_GENERATED_3_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl-l2.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateNzslL2> UpdateNzslL2Truncate;
typedef UpdateAbs<UpdateNzslL2> UpdateNzslL2Abs;
typedef UpdateTruncate<UpdateAbs<UpdateNzslL2> > UpdateNzslL2AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateNzslL2Truncate)
MPI2_TYPE_TRAITS(mf::UpdateNzslL2Abs)
MPI2_TYPE_TRAITS(mf::UpdateNzslL2AbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
