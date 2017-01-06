#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-21.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeL1>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeL1Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL1>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL1Truncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1AbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL1Truncate, mf::SumLoss_NzslLoss_L1Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1AbsTruncate, mf::SumLoss_NzslLoss_L1Loss>;
}

