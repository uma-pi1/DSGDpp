#ifndef  REGISTER_GENERATED_11_H 
#define  REGISTER_GENERATED_11_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-nzl2.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeNzl2> RegularizeNzl2Truncate;
typedef RegularizeAbs<RegularizeNzl2> RegularizeNzl2Abs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeNzl2> > RegularizeNzl2AbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeNzl2Truncate)
MPI2_TYPE_TRAITS(mf::RegularizeNzl2Abs)
MPI2_TYPE_TRAITS(mf::RegularizeNzl2AbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
