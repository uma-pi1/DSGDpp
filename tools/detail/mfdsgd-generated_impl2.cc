#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class SumLoss<NzslLoss, Nzl2Loss>;
template class mf::DsgdJob<mf::UpdateSl, mf::RegularizeSl>;
template class mf::DsgdJob<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate>;
template class mf::DistributedDecayAuto< mf::UpdateNzslL2Truncate, mf::RegularizeNone, mf::SumLoss<NzslLoss, L2Loss> >;
template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone>;
}


