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
#ifndef  REGISTER_GENERATED_1_H 
#define  REGISTER_GENERATED_1_H 

#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/sgd/functions/update-none.h>
#include <mf/sgd/functions/update-abs.h>
#include <mf/sgd/functions/update-truncate.h>

namespace mf {
typedef UpdateTruncate<UpdateNone> UpdateNoneTruncate;
typedef UpdateAbs<UpdateNone> UpdateNoneAbs;
typedef UpdateTruncate<UpdateAbs<UpdateNone> > UpdateNoneAbsTruncate;
}

MPI2_TYPE_TRAITS(mf::UpdateNoneTruncate)
MPI2_TYPE_TRAITS(mf::UpdateNoneAbs)
MPI2_TYPE_TRAITS(mf::UpdateNoneAbsTruncate)

namespace mf {
}


namespace mf {
}



namespace mf {
}

#endif
