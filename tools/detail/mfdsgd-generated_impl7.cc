#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class mf::DistributedDecayAuto< mf::UpdateGkl, mf::RegularizeGkl, mf::GklLoss >;
template class mf::DistributedDecayAuto< mf::UpdateSlTruncate, mf::RegularizeSlTruncate, mf::SlLoss >;
template class mf::DsgdJob<mf::UpdateNzslL2Truncate, mf::RegularizeNone>;
template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2, mf::RegularizeNone, mf::SumLoss<NzslLoss, Nzl2Loss> >;
template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
}


