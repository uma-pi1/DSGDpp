#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-25.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Abs, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
}

