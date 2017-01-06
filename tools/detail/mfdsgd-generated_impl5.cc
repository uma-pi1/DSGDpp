#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class mf::DsgdJob<mf::UpdateGklTruncate, mf::RegularizeGkl>;
template class mf::DistributedDecayAuto< mf::UpdateSl, mf::RegularizeSl, mf::SlLoss >;
template class mf::DistributedDecayAuto< mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate, mf::NzslLoss >;
template class mf::DsgdJob<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone>;
template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2, mf::RegularizeNone, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
}


