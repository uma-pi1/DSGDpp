#ifndef  REGISTER_GENERATED_19_H 
#define  REGISTER_GENERATED_19_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-sl.h>
#include <mf/sgd/functions/regularize-sl.h>
#include <mf/loss/sl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-5.h>
#include <mf/register/register-generated-12.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSl, mf::RegularizeSl, mf::SlLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSlAbs, mf::RegularizeSlAbs, mf::SlLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSlTruncate, mf::RegularizeSl, mf::SlLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSlTruncate, mf::RegularizeSlTruncate, mf::SlLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSlAbsTruncate, mf::RegularizeSlAbs, mf::SlLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateSlAbsTruncate, mf::RegularizeSlAbsTruncate, mf::SlLoss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateSl, mf::RegularizeSl>;
extern template class mf::detail::DsgdTask<mf::UpdateSlAbs, mf::RegularizeSlAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateSlTruncate, mf::RegularizeSl>;
extern template class mf::detail::DsgdTask<mf::UpdateSlTruncate, mf::RegularizeSlTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSl, mf::RegularizeSl, mf::SlLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbs, mf::RegularizeSlAbs, mf::SlLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlTruncate, mf::RegularizeSl, mf::SlLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlTruncate, mf::RegularizeSlTruncate, mf::SlLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbs, mf::SlLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateSlAbsTruncate, mf::RegularizeSlAbsTruncate, mf::SlLoss>;
}

#endif
