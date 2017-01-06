#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class mf::DsgdJob<mf::UpdateGkl, mf::RegularizeGkl>;
template class mf::DsgdJob<mf::UpdateSlTruncate, mf::RegularizeSlTruncate>;
template class mf::DistributedDecayAuto< mf::UpdateNzslTruncate, mf::RegularizeNone, mf::NzslLoss >;
template class mf::DsgdJob<mf::UpdateNzslNzl2, mf::RegularizeNone>;
template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
}


