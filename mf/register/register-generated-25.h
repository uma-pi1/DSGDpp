#ifndef  REGISTER_GENERATED_25_H 
#define  REGISTER_GENERATED_25_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl-nzl2.h>
#include <mf/sgd/functions/regularize-none.h>
#include <mf/loss/nzl2.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-4.h>
#include <mf/register/register-generated-8.h>
#include <mf/register/register-generated-16.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Abs, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_Nzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_Nzl2Loss>;
}

#endif
