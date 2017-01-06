#ifndef  REGISTER_GENERATED_8_H 
#define  REGISTER_GENERATED_8_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-none.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeNone> RegularizeNoneTruncate;
typedef RegularizeAbs<RegularizeNone> RegularizeNoneAbs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeNone> > RegularizeNoneAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeNoneTruncate)
MPI2_TYPE_TRAITS(mf::RegularizeNoneAbs)
MPI2_TYPE_TRAITS(mf::RegularizeNoneAbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
