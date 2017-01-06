#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-24.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzslL2, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslL2Abs, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslL2Truncate, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_L2Loss>;
}

