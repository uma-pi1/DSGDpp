#ifndef  REGISTER_GENERATED_26_H 
#define  REGISTER_GENERATED_26_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-biased-nzsl-nzl2.h>
#include <mf/sgd/functions/regularize-none.h>
#include <mf/loss/biased-nzl2.h>
#include <mf/loss/biased-nzsl.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/register/register-generated-7.h>
#include <mf/register/register-generated-8.h>
#include <mf/register/register-generated-17.h>
#include <mf/sgd/dsgd.h>

namespace mf {
}


namespace mf {
}


namespace mf {
}


MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);
MPI2_TYPE_TRAITS3(mf::detail::DistributedDecayAutoTask, mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss);

namespace mf {
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Abs, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone>;
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbs>;
extern template class mf::detail::DsgdTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Abs, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbs, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
extern template class mf::detail::DistributedDecayAutoTask<mf::UpdateBiasedNzslNzl2AbsTruncate, mf::RegularizeNoneAbsTruncate, mf::SumLoss_BiasedNzslLoss_BiasedNzl2Loss>;
}

#endif
