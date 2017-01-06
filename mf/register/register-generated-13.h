#ifndef  REGISTER_GENERATED_13_H 
#define  REGISTER_GENERATED_13_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-gkl.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeGkl> RegularizeGklTruncate;
typedef RegularizeAbs<RegularizeGkl> RegularizeGklAbs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeGkl> > RegularizeGklAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeGklTruncate)
MPI2_TYPE_TRAITS(mf::RegularizeGklAbs)
MPI2_TYPE_TRAITS(mf::RegularizeGklAbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
