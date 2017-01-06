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
#ifndef MF_AP_DAP_FACTORIZATION_H
#define MF_AP_DAP_FACTORIZATION_H

#include <mf/factorization.h>

namespace mf {

template<typename M1, typename M2, typename M3, typename M4>
bool checkBlockingDap(const DistributedMatrix<M1>& v, const DistributedMatrix<M2>& w,
		const DistributedMatrix<M3>& h, const DistributedMatrix<M4>& vc) {
	if (v.size1() != vc.size1()) return false;
	if (v.size2() != vc.size2()) return false;
	if (v.blocks2() != 1 || vc.blocks1() != 1) return false;
	if (v.blocks1() != vc.blocks2()) return false;

	if (v.blocks1() != w.blocks1() || w.blocks2() != 1) return false;
	if (vc.blocks2() != h.blocks2() || h.blocks1() != 1) return false;
	if (v.blockOffsets1() != w.blockOffsets1()) return false;
	if (vc.blockOffsets2() != h.blockOffsets2()) return false;

	// TOOD: check other requirements (see DapFactorizationData)

	return true;
}

/** Data structure that describes the data, starting point, and result of an
 * DAP (distributed alternating projections) job.
 *
 * Let w be the number of ranks. Let t be the number of tasks per rank. Set d=w*t.
 * The distribution of the matrices satisfies:
 * (1) V is blocked d x 1,
 * (2) W is conformingly (to V) blocked d x 1,
 * (3) VC is blocked 1 x d,
 * (4) H is conformingly (to VC) blocked 1 x d,
 * (5) each row block of V and the corresponding block of W are located on a single rank,
 * (6) each column block of VC and the corresponding block of H are located on a single rank,
 * (7) each rank holds exactly t rows of blocks of V,
 * (8) each rank holds exactly t columns of blocks of VC
 */
template<typename Data = double, typename Factor = double>
struct DapFactorizationData : public DistributedFactorizationData<Data,Factor> {
public:
	typedef DistributedFactorizationData<Data,Factor> Base;
	typedef typename Base::V V;
	typedef typename Base::VC VC;
	typedef typename Base::W W;
	typedef typename Base::H H;
	typedef typename Base::DV DV;
	typedef typename Base::DVC DVC;
	typedef typename Base::DW DW;
	typedef typename Base::DH DH;

	DapFactorizationData(
			const DV& dv,
			DW& dw,
			DH& dh, unsigned tasksPerRank, const DVC* dvc)
	: Base(dv, dw, dh, tasksPerRank, dvc) {
		if (!checkBlockingDap(dv, dw, dh, *dvc)) RG_THROW(rg::InvalidArgumentException, "");
	}

	DapFactorizationData(DapFactorizationData<Data,Factor>& o)
	: Base(o) {
	}

	DapFactorizationData(mpi2::SerializationConstructor _)
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


MPI2_SERIALIZATION_CONSTRUCTOR2(mf::DapFactorizationData);

#endif
