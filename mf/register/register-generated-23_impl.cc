#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-23.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeNzl2>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeNzl2Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2Truncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2Abs>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2AbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2Truncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2AbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
}

