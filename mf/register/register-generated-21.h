#ifndef  REGISTER_GENERATED_21_H 
#define  REGISTER_GENERATED_21_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/regularize-l1.h>
#include <mf/loss/l1.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-2.h>
#include <mf/register/register-generated-9.h>
#include <mf/register/register-generated-14.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzsl, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbs, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeL1Truncate, mf::SumLoss_NzslLoss_L1Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeL1AbsTruncate, mf::SumLoss_NzslLoss_L1Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeL1>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeL1Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL1>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeL1Truncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1AbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL1, mf::SumLoss_NzslLoss_L1Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeL1Truncate, mf::SumLoss_NzslLoss_L1Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1Abs, mf::SumLoss_NzslLoss_L1Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeL1AbsTruncate, mf::SumLoss_NzslLoss_L1Loss>;
}

#endif
