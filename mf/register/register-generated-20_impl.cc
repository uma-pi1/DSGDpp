#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-20.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeNone, mf::NzslLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeNoneAbs, mf::NzslLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNone, mf::NzslLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate, mf::NzslLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbs, mf::NzslLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbsTruncate, mf::NzslLoss>;
}

