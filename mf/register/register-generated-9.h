#ifndef  REGISTER_GENERATED_9_H 
#define  REGISTER_GENERATED_9_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-l1.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeL1> RegularizeL1Truncate;
typedef RegularizeAbs<RegularizeL1> RegularizeL1Abs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeL1> > RegularizeL1AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeL1Truncate)
MPI2_TYPE_TRAITS(mf::RegularizeL1Abs)
MPI2_TYPE_TRAITS(mf::RegularizeL1AbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
