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
#ifndef MF_AP_DSGD_FACTORIZATION_H
#define MF_AP_DSGD_FACTORIZATION_H

#include <mf/factorization.h>

namespace mf {

template<typename M1, typename M2, typename M3>
bool checkBlockingDsgd(const DistributedMatrix<M1>& v, const DistributedMatrix<M2>& w,
		const DistributedMatrix<M3>& h) {
	if (v.blocks1() != v.blocks2()) return false; // d-by-d condition
	if (v.blocks1() != w.blocks1() || w.blocks2() != 1) return false;
	if (v.blocks2() != h.blocks2() || h.blocks1() != 1) return false;
	if (v.blockOffsets1() != w.blockOffsets1()) return false;
	if (v.blockOffsets2() != h.blockOffsets2()) return false;

	// TOOD: check other requirements (see DsgdFactorizationData)

	return true;
}

/** Data structure that describes the data, starting point, and result of an
 * DSGD job.
 *
 * Let w be the number of ranks. Let t be the number of tasks per rank. Set d=w*t.
 * The distribution of the matrices satisfies:
 * (1) V is blocked d x d,
 * (2) W is conformingly blocked d x 1,
 * (3) H is conformingly blocked 1 x d,
 * (4) each row of blocks of V and the corresponding block of W are located on a single rank,
 * (5) each rank holds exactly t rows of blocks of V,
 * (6) each rank holds exactly t blocks of H.
 */
template<typename Data = double, typename Factor = double>
struct DsgdFactorizationData : public DistributedFactorizationData<Data,Factor> {
public:
	typedef DistributedFactorizationData<Data,Factor> Base;
	typedef typename Base::V V;
	typedef typename Base::W W;
	typedef typename Base::H H;
	typedef typename Base::DV DV;
	typedef typename Base::DW DW;
	typedef typename Base::DH DH;

	DsgdFactorizationData(
			const DV& dv,
			DW& dw,
			DH& dh, unsigned tasksPerRank = 1)
	: Base(dv, dw, dh, tasksPerRank) {
		if (!checkBlockingDsgd(dv, dw, dh)) RG_THROW(rg::InvalidArgumentException, "");
	}

	DsgdFactorizationData(DsgdFactorizationData<Data,Factor>& o)
	: Base(o) {
	}

	DsgdFactorizationData(mpi2::SerializationConstructor _)
	: Base(mpi2::UNINITIALIZED) {
	};

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<Base>(*this);
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::DsgdFactorizationData);

#endif
