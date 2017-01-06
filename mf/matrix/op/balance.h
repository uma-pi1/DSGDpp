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
#ifndef MF_MATRIX_OP_BALANCE_H
#define MF_MATRIX_OP_BALANCE_H

#include <boost/numeric/ublas/vector.hpp>
#include <util/exception.h>
#include <mf/factorization.h>
#include <mf/sgd/dsgd-factorization.h>
#include <mf/sgd/dsgdpp-factorization.h>
#include <mf/ap/dap-factorization.h>
#include <mf/sgd/asgd-factorization.h>

namespace mf {

enum BalanceType { BALANCE_NONE, BALANCE_L2, BALANCE_NZL2 };
enum BalanceMethod {
	BALANCE_SIMPLE,         /**< Balance using a single constant */
	BALANCE_OPTIMAL         /**< Balance using a different constant for each factor */
};

boost::numeric::ublas::vector<double> balanceSimple(FactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balanceOptimal(FactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balance(FactorizationData<>& data, BalanceType type = BALANCE_L2, BalanceMethod method = BALANCE_SIMPLE);

boost::numeric::ublas::vector<double> balanceSimple(DsgdFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balanceOptimal(DsgdFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balance(DsgdFactorizationData<>& data,
		BalanceType type = BALANCE_L2, BalanceMethod method = BALANCE_SIMPLE);

boost::numeric::ublas::vector<double> balanceSimple(DsgdPpFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balanceOptimal(DsgdPpFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balance(DsgdPpFactorizationData<>& data,
		BalanceType type = BALANCE_L2, BalanceMethod method = BALANCE_SIMPLE);

boost::numeric::ublas::vector<double> balanceSimple(DapFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balanceOptimal(DapFactorizationData<>& data, BalanceType type);
boost::numeric::ublas::vector<double> balance(DapFactorizationData<>& data,
		BalanceType type = BALANCE_L2, BalanceMethod method = BALANCE_SIMPLE);

inline boost::numeric::ublas::vector<double> balance(AsgdFactorizationData<>& data,
		BalanceType type = BALANCE_L2, BalanceMethod method = BALANCE_SIMPLE) {
	boost::numeric::ublas::vector<double> factors;
	if (type == BALANCE_NONE) {
		return factors;
	}
	RG_THROW(rg::NotImplementedException, "balancing for ASGD");
};

}

#endif
