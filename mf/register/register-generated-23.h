#ifndef  REGISTER_GENERATED_23_H 
#define  REGISTER_GENERATED_23_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl.h>
#include <mf/sgd/functions/regularize-nzl2.h>
#include <mf/loss/nzl2.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-2.h>
#include <mf/register/register-generated-11.h>
#include <mf/register/register-generated-16.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzsl, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbs, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslTruncate, mf::RegularizeNzl2Truncate, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2AbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzsl, mf::RegularizeNzl2>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbs, mf::RegularizeNzl2Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2Truncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2Abs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2AbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzsl, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbs, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslTruncate, mf::RegularizeNzl2Truncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2Abs, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslAbsTruncate, mf::RegularizeNzl2AbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
}

#endif
