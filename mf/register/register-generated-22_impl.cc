#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-22.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeL2>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeL2Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL2>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL2Truncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2AbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL2Truncate, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2AbsTruncate, mf::SumLoss_NzslLoss_L2Loss>;
}

