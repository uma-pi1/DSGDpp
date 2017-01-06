#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 

namespace mf {
template class mf::DistributedDecayAuto< mf::UpdateGklTruncate, mf::RegularizeGkl, mf::GklLoss >;
template class mf::DsgdJob<mf::UpdateNzsl, mf::RegularizeNone>;
template class mf::DsgdJob<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate>;
template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss<NzslLoss, Nzl2Loss> >;
}


