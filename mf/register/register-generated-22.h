#ifndef  REGISTER_GENERATED_22_H 
#define  REGISTER_GENERATED_22_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/regularize-l2.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-2.h>
#include <mf/register/register-generated-10.h>
#include <mf/register/register-generated-15.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzsl, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbs, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeL2Truncate, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeL2AbsTruncate, mf::SumLoss_NzslLoss_L2Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeL2>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeL2Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL2>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL2Truncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2AbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL2, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL2Truncate, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2Abs, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL2AbsTruncate, mf::SumLoss_NzslLoss_L2Loss>;
}

#endif
