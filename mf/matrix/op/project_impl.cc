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
#include <mpi2/mpi2.h>

#include <mf/matrix/op/project.h>
#include <mf/matrix/op/nnz.h>

namespace mf {

void projectSubmatrix(const SparseMatrix& V, SparseMatrix& result,
		const std::vector<mf_size_type>& indexes1,
		const std::vector<mf_size_type>& indexes2) {
	// TO SLOW
	// boost::numeric::ublas::indirect_array<std::vector<unsigned> > i1(nrows, rows);
	// boost::numeric::ublas::indirect_array<std::vector<unsigned> > i2(ncols, cols);
	// result = boost::numeric::ublas::project(V, i1, i2);

	// built inverted index for efficiency
	std::vector<mf_size_type> r(V.size1(), -1);
	for (mf_size_type p=0; p<indexes1.size(); p++) {
		r[indexes1[p]] = p;
	}
	std::vector<mf_size_type> c(V.size2(), -1);
	for (mf_size_type p=0; p<indexes2.size(); p++) {
		c[indexes2[p]] = p;
	}

	// iterate over the data
	result.resize(indexes1.size(), indexes2.size(), false);
	const SparseMatrix::index_array_type& index1 = V.index1_data();
	const SparseMatrix::index_array_type& index2 = V.index2_data();
	const SparseMatrix::value_array_type& values = V.value_data();
	for (mf_size_type p=0; p<V.nnz(); p++) { // TODO: could be optimized
		int row = r[index1[p]];
		int col = c[index2[p]];
		if (row >= 0 && col >= 0) {
			result.append_element(row, col, values[p]);
		}
	}
	result.sort();
}

void projectRandomSubmatrix(rg::Random32& random, const SparseMatrix& V,
		ProjectedSparseMatrix& sample, mf_size_type n1, mf_size_type n2) {
	if (n2==0) n2=n1;
	sample.map1 = rg::sample(random, n1, V.size1());
	sample.map2 = rg::sample(random, n2, V.size2());
	sample.size1=V.size1();
	sample.size2=V.size2();
	projectSubmatrix(V, sample.data, sample.map1, sample.map2);
};
/*
void projectNonempty(const SparseMatrix& m, ProjectedSparseMatrix& result) {
	// count the number of nonzero entries in each row/column
	std::vector<mf_size_type> nnz1, nnz2;
	nnz12(m, nnz1, nnz2);

	// compute rows/columns that remain
	result.size1 = m.size1();
	result.size2 = m.size2();
	result.map1.clear();
	result.map1.reserve(m.size1());
	result.map2.clear();
	result.map2.reserve(m.size2());
	for (mf_size_type i=0; i<m.size1(); i++) {
		if (nnz1[i]>0) result.map1.push_back(i);
	}
	for (mf_size_type j=0; j<m.size2(); j++) {
		if (nnz2[j]>0) result.map2.push_back(j);
	}

	// create the output matrix
	projectSubmatrix(m, result.data, result.map1, result.map2);
}

void projectNonempty(SparseMatrix& m) {
	ProjectedSparseMatrix projected;
	projectNonempty(m, projected);
	m = projected.data;
}/**/

void projectFrequent(const SparseMatrix& m, ProjectedSparseMatrix& result, mf_size_type t) {
	// count the number of nonzero entries in each row/column
	std::vector<mf_size_type> nnz1, nnz2;
	mf_size_type nnz12max;
	nnz12(m, nnz1, nnz2,nnz12max);

	// compute rows/columns that remain
	result.size1 = m.size1();
	result.size2 = m.size2();
	result.map1.clear();
	result.map1.reserve(m.size1());
	result.map2.clear();
	result.map2.reserve(m.size2());
	for (mf_size_type i=0; i<m.size1(); i++) {
		if (nnz1[i]>t) result.map1.push_back(i);
	}
	for (mf_size_type j=0; j<m.size2(); j++) {
		if (nnz2[j]>t) result.map2.push_back(j);
	}

	// create the output matrix
	projectSubmatrix(m, result.data, result.map1, result.map2);
}

void projectFrequent(SparseMatrix& m, mf_size_type t) {
	ProjectedSparseMatrix projected;
	projectFrequent(m, projected, t);
	m = projected.data;
}
/*
void projectNonempty(const ProjectedSparseMatrix& m, ProjectedSparseMatrix& result) {
	// count the number of nonzero entries in each row/column
	std::vector<mf_size_type> nnz1, nnz2;
	nnz12(m.data, nnz1, nnz2);

	// compute rows/columns that remain
	result.size1 = m.size1;
	result.size2 = m.size2;
	result.map1.clear();
	result.map1.reserve(m.data.size1());
	result.map2.clear();
	result.map2.reserve(m.data.size2());
	for (mf_size_type i=0; i<m.data.size1(); i++) {
		if (nnz1[i]>0) result.map1.push_back(i);
	}
	for (mf_size_type j=0; j<m.data.size2(); j++) {
		if (nnz2[j]>0) result.map2.push_back(j);
	}

	// create the output matrix
	projectSubmatrix(m.data, result.data, result.map1, result.map2);

	// update the row/column map of the output matrix
	for (mf_size_type i=0; i<result.data.size1(); i++) {
		result.map1[i] = m.map1[result.map1[i]];
	}
	for (mf_size_type j=0; j<result.data.size2(); j++) {
		result.map2[j] = m.map2[result.map2[j]];
	}
}

void projectNonempty(ProjectedSparseMatrix& m) {
	ProjectedSparseMatrix copy = m;
	projectNonempty(copy, m);
}/**/

void projectFrequent(const ProjectedSparseMatrix& m, ProjectedSparseMatrix& result, mf_size_type t) {
	// count the number of nonzero entries in each row/column
	std::vector<mf_size_type> nnz1, nnz2;
	mf_size_type nnz12max;
	nnz12(m.data, nnz1, nnz2,nnz12max);

	// compute rows/columns that remain
	result.size1 = m.size1;
	result.size2 = m.size2;
	result.map1.clear();
	result.map1.reserve(m.data.size1());
	result.map2.clear();
	result.map2.reserve(m.data.size2());
	for (mf_size_type i=0; i<m.data.size1(); i++) {
		if (nnz1[i]>t) result.map1.push_back(i);
	}
	for (mf_size_type j=0; j<m.data.size2(); j++) {
		if (nnz2[j]>t) result.map2.push_back(j);
	}

	// create the output matrix
	projectSubmatrix(m.data, result.data, result.map1, result.map2);

	// update the row/column map of the output matrix
	for (mf_size_type i=0; i<result.data.size1(); i++) {
		result.map1[i] = m.map1[result.map1[i]];
	}
	for (mf_size_type j=0; j<result.data.size2(); j++) {
		result.map2[j] = m.map2[result.map2[j]];
	}
}

void projectFrequent(ProjectedSparseMatrix& m, mf_size_type t) {
	ProjectedSparseMatrix copy = m;
	projectFrequent(copy, m, t);
}

void splitIndexes(const std::vector<mf_size_type>& indexes,
		const std::vector<mf_size_type>& blockOffsets,
		std::vector<std::vector<mf_size_type> >& blockIndexes) {
	// initialize
	mf_size_type n = indexes.size();        // number of indexes
	mf_size_type nb = blockOffsets.size();  // number of blocks
	blockIndexes.resize(nb);

	// go
	mf_size_type i = 0;                     			// next index position
	for (mf_size_type ib = 0; ib<nb; ib++) {    			// current block
		mf_size_type ibo = blockOffsets[ib]; 		// current block offset

		blockIndexes[ib].clear();
		mf_size_type max = ib < nb-1 ? blockOffsets[ib+1] : indexes[n-1]+1;
		mf_size_type index;
		while (i<n && (index = indexes[i]) < max) {
			blockIndexes[ib].push_back(index-ibo);
			i++;
		}
	}
}



}
