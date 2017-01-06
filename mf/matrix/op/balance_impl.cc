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
#include <util/exception.h>

#include <mf/logger.h>
#include <mf/matrix/op/balance.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzl2.h>
#include <mf/matrix/op/scale.h>
#include <mf/matrix/op/sums.h>

namespace mf {

boost::numeric::ublas::vector<double> balanceSimple(FactorizationData<>& data, BalanceType type){
	if (type == BALANCE_NONE) return boost::numeric::ublas::vector<double>(1, 1);

	double wFactor, hFactor, regW, regH;

	if (type == BALANCE_L2) {
		regW = l2(data.w);
		regH = l2(data.h);
	} else {
		regW = nzl2(data.w, *data.nnz1, data.nnz1offset);
		regH = nzl2(data.h, *data.nnz2, data.nnz2offset);
	}

	wFactor = sqrt( sqrt(regH/regW) ); // the inner square root is the x that minimizes of x*l2w + 1/x*l2h
	if (std::isnan(wFactor)) {
		LOG4CXX_INFO(detail::logger, "Invalid multiplier in balancing (regW=" << regW << ", regH=" << regH << "); replacing factor matrices by 0 matrices");
		wFactor = 0;
		hFactor = 0;
	} else {
		hFactor = 1./wFactor;
	}

	data.w *= wFactor;
	data.h *= hFactor;

	return boost::numeric::ublas::vector<double>(1, wFactor);
}

boost::numeric::ublas::vector<double> balanceOptimal(FactorizationData<>& data, BalanceType type) {
	if (type == BALANCE_NONE) return boost::numeric::ublas::vector<double>(data.r, 1);

	boost::numeric::ublas::vector<double> wFactor(data.r), hFactor(data.r), regW, regH;

	if (type==BALANCE_L2) {
		regW = squaredSums2(data.w);
		regH = squaredSums1(data.h);
	} else {
		regW = nzl2SquaredSums2(data.w,*data.nnz1, data.nnz1offset);
		regH = nzl2SquaredSums1(data.h,*data.nnz2, data.nnz2offset);
	}

	for (mf_size_type k=0; k<data.r; k++) {
		wFactor[k] = sqrt( sqrt(regH[k] / regW[k]) );
		if (std::isnan(wFactor[k])) {
			LOG4CXX_INFO(detail::logger, "Invalid multiplier in balancing (k=" << k << ", regW=" << regW[k] << ", regH=" << regH[k] << "); replacing factor " << k << " by 0");
			wFactor[k] = 0;
			hFactor[k] = 0;
		} else {
			hFactor[k] = 1./wFactor[k];
		}
	}
	mult2(data.w, wFactor);
	mult1(data.h, hFactor);

	return wFactor;
}

boost::numeric::ublas::vector<double> balance(FactorizationData<>& data, BalanceType type, BalanceMethod method) {
	boost::numeric::ublas::vector<double> factors;
	switch (method) {
	case BALANCE_SIMPLE:
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Starting simple balancing");
		factors = balanceSimple(data, type);
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Balancing factor (W)" << factors[0]);
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Finished simple balancing");
		break;
	case BALANCE_OPTIMAL:
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Starting optimal balancing");
		factors = balanceOptimal(data, type);
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Balancing factors (W): " << factors);
		if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Finished optimal balancing");
		break;
	}

	return factors;
}

namespace detail {
	template<typename FD>
	boost::numeric::ublas::vector<double> balanceSimple(FD& data, BalanceType type) {
		if (type == BALANCE_NONE) return boost::numeric::ublas::vector<double>(1, 1);

		double wFactor, hFactor, regW, regH;

		if (type == BALANCE_L2) {
			regW = l2(data.dw, data.tasksPerRank);
			regH = l2(data.dh, data.tasksPerRank);
		} else {
			regW = nzl2(data.dw, data.nnz1name, data.tasksPerRank);
			regH = nzl2(data.dh, data.nnz2name, data.tasksPerRank);
		}

		wFactor = sqrt( sqrt(regH/regW) ); // the inner square root is the x that minimizes of x*l2w + 1/x*l2h
		if (std::isnan(wFactor)) {
			LOG4CXX_INFO(detail::logger, "Invalid multiplier in balancing (regW=" << regW << ", regH=" << regH << "); replacing factor matrices by 0 matrices");
			wFactor = 0;
			hFactor = 0;
		} else {
			hFactor = 1./wFactor;
		}

		mult(data.dw, wFactor, data.tasksPerRank);
		mult(data.dh, hFactor, data.tasksPerRank);

		return boost::numeric::ublas::vector<double>(1, wFactor);
	}

	template<typename FD>
	boost::numeric::ublas::vector<double> balanceOptimal(FD& data, BalanceType type) {
		mf_size_type r = data.dw.size2();
		if (type == BALANCE_NONE) return boost::numeric::ublas::vector<double>(r, 1);

		boost::numeric::ublas::vector<double> wFactor(r), hFactor(r), regW, regH;

		if (type==BALANCE_L2) {
			regW = squaredSums2(data.dw, data.tasksPerRank);
			regH = squaredSums1(data.dh, data.tasksPerRank);
		} else {
			//RG_THROW(rg::NotImplementedException, "distributed balancing with Nzl2 is implemented now!");
			regW = nzl2SquaredSums2(data.dw, data.nnz1name, data.tasksPerRank);
			regH = nzl2SquaredSums1(data.dh, data.nnz2name, data.tasksPerRank);
		}

		for (mf_size_type k=0; k<r; k++) {
			wFactor[k] = sqrt( sqrt(regH[k] / regW[k]) );
			if (std::isnan(wFactor[k])) {
				LOG4CXX_INFO(detail::logger, "Invalid multiplier in balancing (k=" << k << ", regW=" << regW[k] << ", regH=" << regH[k] << "); replacing factor " << k << " by 0");
				wFactor[k] = 0;
				hFactor[k] = 0;
			} else {
				hFactor[k] = 1./wFactor[k];
			}
		}
		mult2(data.dw, wFactor, data.tasksPerRank);
		mult1(data.dh, hFactor, data.tasksPerRank);

		return wFactor;
	}

	template<typename FD>
	boost::numeric::ublas::vector<double> balance(FD& data, BalanceType type, BalanceMethod method) {
		boost::numeric::ublas::vector<double> factors;
		switch (method) {
		case BALANCE_SIMPLE:
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Starting simple balancing");
			factors = balanceSimple(data, type);
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Balancing factor (W): " << factors[0]);
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Finished simple balancing");
			break;
		case BALANCE_OPTIMAL:
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Starting optimal balancing");
			factors = balanceOptimal(data, type);
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Balancing factors (W): " << factors);
			if (type != BALANCE_NONE) LOG4CXX_INFO(mf::detail::logger, "Finished optimal balancing");
			break;
		}

		return factors;
	}
}

boost::numeric::ublas::vector<double> balanceSimple(DsgdFactorizationData<>& data, BalanceType type) {
	return detail::balanceSimple(data, type);
}
boost::numeric::ublas::vector<double> balanceOptimal(DsgdFactorizationData<>& data, BalanceType type) {
	return detail::balanceOptimal(data, type);
}

boost::numeric::ublas::vector<double> balance(DsgdFactorizationData<>& data,
		BalanceType type, BalanceMethod method) {
	return detail::balance(data, type, method);
}

boost::numeric::ublas::vector<double> balanceSimple(DsgdPpFactorizationData<>& data, BalanceType type) {
	return detail::balanceSimple(data, type);
}
boost::numeric::ublas::vector<double> balanceOptimal(DsgdPpFactorizationData<>& data, BalanceType type) {
	return detail::balanceOptimal(data, type);
}

boost::numeric::ublas::vector<double> balance(DsgdPpFactorizationData<>& data,
		BalanceType type, BalanceMethod method) {
	return detail::balance(data, type, method);
}


boost::numeric::ublas::vector<double> balanceSimple(DapFactorizationData<>& data, BalanceType type) {
	return detail::balanceSimple(data, type);
}
boost::numeric::ublas::vector<double> balanceOptimal(DapFactorizationData<>& data, BalanceType type) {
	return detail::balanceOptimal(data, type);
}

boost::numeric::ublas::vector<double> balance(DapFactorizationData<>& data,
		BalanceType type, BalanceMethod method) {
	return detail::balance(data, type, method);
}


}
