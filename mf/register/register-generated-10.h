#ifndef  REGISTER_GENERATED_10_H 
#define  REGISTER_GENERATED_10_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-l2.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeL2> RegularizeL2Truncate;
typedef RegularizeAbs<RegularizeL2> RegularizeL2Abs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeL2> > RegularizeL2AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeL2Truncate)
MPI2_TYPE_TRAITS(mf::RegularizeL2Abs)
MPI2_TYPE_TRAITS(mf::RegularizeL2AbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
