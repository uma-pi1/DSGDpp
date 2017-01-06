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
/** \file
 * Describing input and output of factorization jobs.
 */

#ifndef MF_FACTORIZATION_H
#define MF_FACTORIZATION_H

//#include <mf/sgd/decay_auto.h>
#include <util/io.h>

#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/op/nnz.h>

namespace mf {

template<typename M1, typename M2, typename M3>
bool checkConformity(const M1& v, const M2& w, const M3& h) {
	if (v.size1() != w.size1()) return false;
	if (v.size2() != h.size2()) return false;
	if (w.size2() != h.size1()) return false;
	return true;
}

template<typename M1, typename M2, typename M3, typename M4>
bool checkConformity(const M1& v, const M2& w, const M3& h, const M4* vc) {
	if (!checkConformity(v, w, h)) return false;
	if (vc != NULL && v.size1() != vc->size1()) return false;
	if (vc != NULL && v.size2() != vc->size2()) return false;
	return true;
}

/** Data structure that describes the data, starting point, and result of a factorization job
 * (for methods: SGD, PSGD, or AP).
 *
 * The "tasks" member variable is only used for PSGD; it determines the number of parallel tasks.
 * The "vc" member variables is only used for AP; it contains a column-major version of the data.
 *
 * @tparam Data element type of data matrix
 * @tparam Factor element type of factor matrices
 */
template<typename Data = double, typename Factor = double>
struct FactorizationData {
	typedef boost::numeric::ublas::coordinate_matrix<Data, boost::numeric::ublas::row_major> V;
	typedef boost::numeric::ublas::coordinate_matrix<Data, boost::numeric::ublas::column_major> VC;
	typedef boost::numeric::ublas::matrix<Factor, boost::numeric::ublas::row_major> W;
	typedef boost::numeric::ublas::matrix<Factor, boost::numeric::ublas::column_major> H;

	FactorizationData(const V& v, W& w, H& h,
			const std::vector<mf_size_type>& nnz1, mf_size_type nnz1offset,
			const std::vector<mf_size_type>& nnz2, mf_size_type nnz2offset, mf_size_type nnz12max,
			int tasks=1, VC* vc = NULL)
	: v(v), vc(vc), w(w), h(h),
	  vIndex1(v.index1_data()), vIndex2(v.index2_data()), vValues(v.value_data()),
	  wValues(w.data()), hValues(h.data()),
	  nnz(v.nnz()), m(v.size1()), n(v.size2()), r(w.size2()),
	  nnz1(&nnz1), nnz1offset(nnz1offset), nnz2(&nnz2), nnz2offset(nnz2offset) ,nnz12max(nnz12max),
	  tasks(tasks)
	{
		if (!checkConformity(v, w, h, vc)) RG_THROW(rg::InvalidArgumentException, "");
	}

	FactorizationData(const V& v, W& w, H& h, int tasks=1, VC* vc = NULL)
	: v(v), vc(vc), w(w), h(h),
	  vIndex1(v.index1_data()), vIndex2(v.index2_data()), vValues(v.value_data()),
	  wValues(w.data()), hValues(h.data()),
	  nnz(v.nnz()), m(v.size1()), n(v.size2()), r(w.size2()),
	  nnz1(new std::vector<mf_size_type>(v.size1())), nnz1offset(0),
	  nnz2(new std::vector<mf_size_type>(v.size2())), nnz2offset(0),
	  tasks(tasks)
	{
		if (!checkConformity(v, w, h, vc)) RG_THROW(rg::InvalidArgumentException, "");
		nnz12(v, *const_cast<std::vector<mf_size_type> *>(nnz1), *const_cast<std::vector<mf_size_type> *>(nnz2), nnz12max);
	}

	~FactorizationData() {
		// TODO clean up nnz1, nnz2 if created automatically
	}

	/** Input matrix, row-major */
	const V& v;

	/** Input matrix, column-major (AP only) */
	const VC* vc;

	/** Row parameters */
	W& w;

	/** Column parameters */
	H& h;

	/** Array of row numbers from V */
	const typename V::index_array_type& vIndex1;

	/** Array of column numbers from V */
	const typename V::index_array_type& vIndex2;

	/** Array of values from V */
	const typename V::value_array_type& vValues;

	/** Array of values from w (row major) */
	typename W::array_type& wValues;

	/** Array of values from h (column major) */
	typename H::array_type& hValues;

	/** Number of nonzero entries in v */
	const mf_size_type nnz;

	/** Number of rows of v and w */
	const mf_size_type m;

	/** Number of columns of v and h */
	const mf_size_type n;

	/** Number of rows of w / columns of h. Rank of the factorization */
	const mf_size_type r;

	/** Number of nonzero entries in each row of v */
	const std::vector<mf_size_type>* const nnz1;
	const mf_size_type nnz1offset;

	/** Number of nonzero entries in each column of v */
	const std::vector<mf_size_type>* const nnz2;
	const mf_size_type nnz2offset;

	/** Maximum number of nonzero entries in columns or rows of v */
	mf_size_type nnz12max;

	/** Number of parallel tasks (PSGD only) */
	int tasks;
};

/** Data structure that describes the data, starting point, and result of a
 * distributed factorization job. Subclasses are used for the different methods.
 *
 * @tparam Data element type of data matrix
 * @tparam Factor element type of factor matrices
 */
template<typename Data = double, typename Factor = double>
struct DistributedFactorizationData {
public:
	typedef boost::numeric::ublas::coordinate_matrix<Data, boost::numeric::ublas::row_major> V;
	typedef boost::numeric::ublas::coordinate_matrix<Data, boost::numeric::ublas::column_major> VC;
	typedef boost::numeric::ublas::matrix<Factor, boost::numeric::ublas::row_major> W;
	typedef boost::numeric::ublas::matrix<Factor, boost::numeric::ublas::column_major> H;
	typedef DistributedMatrix<V> DV;
	typedef DistributedMatrix<VC> DVC;
	typedef DistributedMatrix<W> DW;
	typedef DistributedMatrix<H> DH;

	const DV dv;
	const DVC* dvc;
	DW dw;
	DH dh;
	mf_size_type nnz;
	unsigned tasksPerRank;
	std::string nnz1name; // nnz data will be / needs to be replicated to each node
	std::string nnz2name;
	mf_size_type nnz12max;// Maximum number of nonzero entries in columns or rows of v

	const std::vector<mf_size_type>& nnz1() const {
		return *mpi2::env().get<std::vector<mf_size_type> >(nnz1name);
	}
	const std::vector<mf_size_type>& nnz2() const {
		return *mpi2::env().get<std::vector<mf_size_type> >(nnz2name);
	}

protected:
	DistributedFactorizationData(
			const DV& dv,
			DW& dw,
			DH& dh, unsigned tasksPerRank = 1, const DVC* dvc = NULL)
	: dv(dv), dvc(dvc), dw(dw), dh(dh), tasksPerRank(tasksPerRank) {
		checkConformity(dv, dw, dh, dvc);
		nnz1name = rg::paste(dv.name(), "_nnz1");
		nnz2name = rg::paste(dv.name(), "_nnz2");
		init();
	}

	DistributedFactorizationData(DistributedFactorizationData<Data,Factor>& o)
	: dv(o.dv), dvc(o.dvc), dw(o.dw), dh(o.dh), nnz(o.nnz), tasksPerRank(o.tasksPerRank),
	  nnz1name(o.nnz1name), nnz2name(o.nnz2name),nnz12max(o.nnz12max) {
	}

	DistributedFactorizationData(mpi2::SerializationConstructor _)
	: dv(mpi2::UNINITIALIZED), dvc(NULL), dw(mpi2::UNINITIALIZED), dh(mpi2::UNINITIALIZED) {
	}

	void init() {
		nnz = mf::nnz(dv, tasksPerRank);
		std::vector<mf_size_type> nnz1, nnz2;
		mf::nnz12(dv, nnz1, nnz2, nnz12max, tasksPerRank);
		mpi2::createCopyAll(nnz1name, nnz1);
		mpi2::createCopyAll(nnz2name, nnz2);
	}

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & const_cast<DV&>(dv);
		ar & const_cast<DVC* &>(dvc);
		ar & dw;
		ar & dh;
		ar & nnz;
		ar & tasksPerRank;
		ar & nnz1name;
		ar & nnz2name;
		ar & nnz12max;
	}

};

}

#endif
