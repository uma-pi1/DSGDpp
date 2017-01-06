#ifndef  REGISTER_GENERATED_18_H 
#define  REGISTER_GENERATED_18_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-gkl.h>
#include <mf/sgd/functions/regularize-gkl.h>
#include <mf/loss/gkl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-6.h>
#include <mf/register/register-generated-13.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGkl, mf::RegularizeGkl, mf::GklLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGklAbs, mf::RegularizeGklAbs, mf::GklLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGklTruncate, mf::RegularizeGkl, mf::GklLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGklTruncate, mf::RegularizeGklTruncate, mf::GklLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGklAbsTruncate, mf::RegularizeGklAbs, mf::GklLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateGklAbsTruncate, mf::RegularizeGklAbsTruncate, mf::GklLoss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateGkl, mf::RegularizeGkl>;
extern template class mf::detail::DsgdTask<mf::UpdateGklAbs, mf::RegularizeGklAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateGklTruncate, mf::RegularizeGkl>;
extern template class mf::detail::DsgdTask<mf::UpdateGklTruncate, mf::RegularizeGklTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGkl, mf::RegularizeGkl, mf::GklLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbs, mf::RegularizeGklAbs, mf::GklLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklTruncate, mf::RegularizeGkl, mf::GklLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklTruncate, mf::RegularizeGklTruncate, mf::GklLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbs, mf::GklLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateGklAbsTruncate, mf::RegularizeGklAbsTruncate, mf::GklLoss>;
}

#endif
