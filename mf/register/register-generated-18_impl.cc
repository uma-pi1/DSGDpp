#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/register/register-generated-18.h>

namespace mf {
template class mf::detail::DsgdTask<mf::UpdateGkl, mf::RegularizeGkl>;
template class mf::detail::DsgdTask<mf::UpdateGklAbs, mf::RegularizeGklAbs>;
template class mf::detail::DsgdTask<mf::UpdateGklTruncate, mf::RegularizeGkl>;
template class mf::detail::DsgdTask<mf::UpdateGklTruncate, mf::RegularizeGklTruncate>;
template class mf::detail::DsgdTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbs>;
template class mf::detail::DsgdTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbsTruncate>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGkl, mf::RegularizeGkl, mf::GklLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbs, mf::RegularizeGklAbs, mf::GklLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklTruncate, mf::RegularizeGkl, mf::GklLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklTruncate, mf::RegularizeGklTruncate, mf::GklLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbs, mf::GklLoss>;
template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbsTruncate, mf::GklLoss>;
}

