#ifndef  REGISTER_GENERATED_5_H 
#define  REGISTER_GENERATED_5_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-sl.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateSl> UpdateSlTruncate;
typedef UpdateAbs<UpdateSl> UpdateSlAbs;
typedef UpdateTruncate<UpdateAbs<UpdateSl> > UpdateSlAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateSlTruncate)
MPI2_TYPE_TRAITS(mf::UpdateSlAbs)
MPI2_TYPE_TRAITS(mf::UpdateSlAbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
