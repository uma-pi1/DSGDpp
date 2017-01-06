//    Copyright 2017 Rainer Gemulla
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef MFDSGD_GENERATED_H 
#define MFDSGD_GENERATED_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 

namespace mf {
typedef UpdateTruncate<UpdateNone> UpdateNoneTruncate;
typedef UpdateTruncate<UpdateNzsl> UpdateNzslTruncate;
typedef UpdateTruncate<UpdateNzslL2> UpdateNzslL2Truncate;
typedef UpdateTruncate<UpdateNzslNzl2> UpdateNzslNzl2Truncate;
typedef UpdateTruncate<UpdateBiasedNzslNzl2> UpdateBiasedNzslNzl2Truncate;
typedef UpdateTruncate<UpdateSl> UpdateSlTruncate;
typedef UpdateTruncate<UpdateGkl> UpdateGklTruncate;
}

namespace mf {
typedef RegularizeTruncate<RegularizeNone> RegularizeNoneTruncate;
typedef RegularizeTruncate<RegularizeNzl2> RegularizeNzl2Truncate;
typedef RegularizeTruncate<RegularizeSl> RegularizeSlTruncate;
typedef RegularizeTruncate<RegularizeGkl> RegularizeGklTruncate;
}

namespace mf {
typedef SumLoss<NzslLoss, L2Loss> SumLoss_NzslLoss_L2Loss;
typedef SumLoss<NzslLoss, Nzl2Loss> SumLoss_NzslLoss_Nzl2Loss;
typedef SumLoss<BiasedNzslLoss, BiasedNzl2Loss> SumLoss_BiasedNzslLoss_BiasedNzl2Loss;
}

namespace mf {
extern template class SumLoss<NzslLoss, L2Loss>;
extern template class SumLoss<NzslLoss, Nzl2Loss>;
extern template class SumLoss<BiasedNzslLoss, BiasedNzl2Loss>;
extern template class mf::DsgdJob<mf::UpdateGkl, mf::RegularizeGkl>;
extern template class mf::DsgdJob<mf::UpdateGklTruncate, mf::RegularizeGkl>;
extern template class mf::DsgdJob<mf::UpdateGklTruncate, mf::RegularizeGklTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateGkl, mf::RegularizeGkl, mf::GklLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateGklTruncate, mf::RegularizeGkl, mf::GklLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateGklTruncate, mf::RegularizeGklTruncate, mf::GklLoss >;
extern template class mf::DsgdJob<mf::UpdateSl, mf::RegularizeSl>;
extern template class mf::DsgdJob<mf::UpdateSlTruncate, mf::RegularizeSl>;
extern template class mf::DsgdJob<mf::UpdateSlTruncate, mf::RegularizeSlTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateSl, mf::RegularizeSl, mf::SlLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateSlTruncate, mf::RegularizeSl, mf::SlLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateSlTruncate, mf::RegularizeSlTruncate, mf::SlLoss >;
extern template class mf::DsgdJob<mf::UpdateNzsl, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslTruncate, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateNzsl, mf::RegularizeNone, mf::NzslLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslTruncate, mf::RegularizeNone, mf::NzslLoss >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslTruncate, mf::RegularizeNoneTruncate, mf::NzslLoss >;
extern template class mf::DsgdJob<mf::UpdateNzslL2, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslL2Truncate, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslL2, mf::RegularizeNone, mf::SumLoss<NzslLoss, L2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslL2Truncate, mf::RegularizeNone, mf::SumLoss<NzslLoss, L2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslL2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<NzslLoss, L2Loss> >;
extern template class mf::DsgdJob<mf::UpdateNzslNzl2, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslNzl2Truncate, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2, mf::RegularizeNone, mf::SumLoss<NzslLoss, Nzl2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss<NzslLoss, Nzl2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<NzslLoss, Nzl2Loss> >;
extern template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone>;
extern template class mf::DsgdJob<mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate>;
extern template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2, mf::RegularizeNone, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNone, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
extern template class mf::DistributedDecayAuto< mf::UpdateBiasedNzslNzl2Truncate, mf::RegularizeNoneTruncate, mf::SumLoss<BiasedNzslLoss, BiasedNzl2Loss> >;
}

bool runArgs(Args& args);

#endif
