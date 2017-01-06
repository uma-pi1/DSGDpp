#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class SumLoss<NzslLoss, L2Loss>;
template class mf::DistributedDecayAuto< mf::UpdateGklTruncate, mf::RegularizeGklTruncate, mf::GklLoss >;
template class mf::DsgdJob<mf::UpdateNzslTruncate, mf::RegularizeNone>;
template class mf::DistributedDecayAuto< mf::UpdateNzslL2, mf::RegularizeNone, mf::SumLoss<NzslLoss, L2Loss> >;
template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<NzslLoss, Nzl2Loss> >;
}


