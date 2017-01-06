#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-19.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateSl, mf::RegularizeSl>;
template class mf::detail::DsgdTask<mf::UpdateSlAbs, mf::RegularizeSlAbs>;
template class mf::detail::DsgdTask<mf::UpdateSlTruncate, mf::RegularizeSl>;
template class mf::detail::DsgdTask<mf::UpdateSlTruncate, mf::RegularizeSlTruncate>;
template class mf::detail::DsgdTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbs>;
template class mf::detail::DsgdTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSl, mf::RegularizeSl, mf::SlLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbs, mf::RegularizeSlAbs, mf::SlLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlTruncate, mf::RegularizeSl, mf::SlLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlTruncate, mf::RegularizeSlTruncate, mf::SlLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbs, mf::SlLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbsTruncate, mf::SlLoss>;
}

