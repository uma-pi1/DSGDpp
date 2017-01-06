#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-26.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Abs, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone>;
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbs>;
template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
}

