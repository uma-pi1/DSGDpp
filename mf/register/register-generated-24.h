#ifndef  REGISTER_GENERATED_24_H 
#define  REGISTER_GENERATED_24_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-nzsl-l2.h>
#include <mf/sgd/functions/regularize-none.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-3.h>
#include <mf/register/register-generated-8.h>
#include <mf/register/register-generated-15.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_L2Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2Abs, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2Truncate, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Abs, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Truncate, mf::RegularizeNone, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_NzslLoss_L2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateNzslL2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_NzslLoss_L2Loss>;
}

#endif
