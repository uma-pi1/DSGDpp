#ifndef  REGISTER_GENERATED_6_H 
#define  REGISTER_GENERATED_6_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-gkl.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateGkl> UpdateGklTruncate;
typedef UpdateAbs<UpdateGkl> UpdateGklAbs;
typedef UpdateTruncate<UpdateAbs<UpdateGkl> > UpdateGklAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateGklTruncate)
MPI2_TYPE_TRAITS(mf::UpdateGklAbs)
MPI2_TYPE_TRAITS(mf::UpdateGklAbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
