#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class mf::DsgdJob<mf::UpdateGklTruncate, mf::RegularizeGklTruncate>;
template class mf::DistributedDecayAuto< mf::UpdateSlTruncate, mf::RegularizeSl, mf::SlLoss >;
template class mf::DsgdJob<mf::UpdateNzslL2, mf::RegularizeNone>;
template class mf::DsgdJob<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
}


