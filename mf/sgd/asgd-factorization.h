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
#ifndef MF_AP_ASGD_FACTORIZATION_H
#define MF_AP_ASGD_FACTORIZATION_H

#include <mf/factorization.h>

namespace mf {

template<typename M1, typename M2, typename M3>
bool checkBlockingAsgd(const DistributedMatrix<M1>& v, const DistributedMatrix<M2>& w,
		const DistributedMatrix<M3>& h) {
	if (v.blocks2() != 1) return false;
	if (v.blocks1() != w.blocks1() || w.blocks2() != 1) return false;
	if (v.blocks1() != h.blocks2() || h.blocks1() != 1) return false;
	if (v.blockOffsets1() != w.blockOffsets1()) return false;

	// TOOD: check other requirements (see AsgdFactorizationData)

	return true;
}

/** Data structure that describes the data, starting point, and result of an
 * ASGD job.
 *
 * Let w be the number of ranks.
 * The distribution of the matrices satisfies:
 * (1) V is blocked w x 1,
 * (2) W is conformingly blocked w x 1,
 * (3) H is blocked 1 x w,
 * (4) each row of blocks of V and the corresponding block of W are located on a single rank,
 * (5) each rank holds exactly 1 rows of blocks of V,
 * (6) each rank holds exactly 1 blocks of H.
 * (7) hWorkName is a variable name that refers to an unblocked version of H stored at all ranks
 */
template<typename Data = double, typename Factor = double>
struct AsgdFactorizationData : public DistributedFactorizationData<Data,Factor> {
public:
	typedef DistributedFactorizationData<Data,Factor> Base;
	typedef typename Base::V V;
	typedef typename Base::W W;
	typedef typename Base::H H;
	typedef typename Base::DV DV;
	typedef typename Base::DW DW;
	typedef typename Base::DH DH;

	AsgdFactorizationData(
			const DV& dv,
			DW& dw,
			DH& dh, unsigned tasksPerRank = 1)
	: Base(dv, dw, dh, tasksPerRank) {
		if (!checkBlockingAsgd(dv, dw, dh)) RG_THROW(rg::InvalidArgumentException,
				"Incorrect blocking of data for ASGD");
		hWorkName = "asgd_h_work";
	}

	AsgdFactorizationData(DsgdFactorizationData<Data,Factor>& o)
	: Base(o), hWorkName(o.hUnblockedName) {
	}

	AsgdFactorizationData(mpi2::SerializationConstructor _)
	: Base(mpi2::UNINITIALIZED) {
	};

	std::string hWorkName;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<Base>(*this);
		ar & hWorkName;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::AsgdFactorizationData);

#endif
