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
#ifndef MF_AP_GNMF_H
#define MF_AP_GNMF_H


#include <mf/factorization.h>
#include <mf/trace.h>
#include<mf/matrix/op/balance.h>

namespace mf {


/** Factorizes the given matrix by minimizing the squared loss. If the initial factors are
 * nonnegative, the resulting factors will be as well.
 *
 * @param data factorization data (data.vc must not be null)
 * @param epochs how many epochs to run. Each epoch performs a single scan through the data matrix
 * and updates either w or h
 * @param trace a trace that will be filled with information about the progress of the algorithm
 */
void gnmf(FactorizationData<>& data, unsigned epochs, Trace& trace, BalanceType type = BALANCE_NONE, BalanceMethod method = BALANCE_SIMPLE, FactorizationData<>* testData=NULL);


}

#endif


