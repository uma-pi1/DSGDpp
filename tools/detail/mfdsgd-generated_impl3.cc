#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class SumLoss<BiasedNzslLoss, BiasedNzl2Loss>;
template class mf::DsgdJob<mf::UpdateSlTruncate, mf::RegularizeSl>;
template class mf::DistributedDecayAuto< mf::UpdateNzsl, mf::RegularizeNone, mf::NzslLoss >;
template class mf::DistributedDecayAuto< mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<NzslLoss, L2Loss> >;
template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone>;
}


