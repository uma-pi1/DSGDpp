#ifndef  REGISTER_GENERATED_12_H 
#define  REGISTER_GENERATED_12_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/regularize-sl.h>
#include <mf/sgd/functions/regularize-abs.h>
#include <mf/sgd/functions/regularize-truncate.h>

namespace mf {
}


namespace mf {
typedef RegularizeTruncate<RegularizeSl> RegularizeSlTruncate;
typedef RegularizeAbs<RegularizeSl> RegularizeSlAbs;
typedef RegularizeTruncate<RegularizeAbs<RegularizeSl> > RegularizeSlAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::RegularizeSlTruncate)
MPI2_TYPE_TRAITS(mf::RegularizeSlAbs)
MPI2_TYPE_TRAITS(mf::RegularizeSlAbsTruncate)

namespace mf {
}



namespace mf {
}

#endif
