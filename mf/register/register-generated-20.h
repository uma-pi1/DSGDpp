#ifndef  REGISTER_GENERATED_20_H 
#define  REGISTER_GENERATED_20_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/regularize-none.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-2.h>
#include <mf/register/register-generated-8.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzsl, mf::RegularizeNone, mf::NzslLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbs, mf::RegularizeNoneAbs, mf::NzslLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeNone, mf::NzslLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate, mf::NzslLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbs, mf::NzslLoss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbsTruncate, mf::NzslLoss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeNone, mf::NzslLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeNoneAbs, mf::NzslLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNone, mf::NzslLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate, mf::NzslLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbs, mf::NzslLoss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNoneAbsTruncate, mf::NzslLoss>;
}

#endif
