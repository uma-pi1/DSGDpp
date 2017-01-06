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
#ifndef MF_MATRIX_OP_COPY
#define MF_MATRIX_OP_COPY

namespace mf {

/** Copies a row-major sparse matrix into a column-major sparse matrix. */
inline void copyCm(const SparseMatrix& m, SparseMatrixCM& mc) {
	mc.clear();
	mc.resize(m.size1(), m.size2(), false);
	mc.reserve(m.nnz(), false);
	std::memmove((void *)mc.index2_data().begin(), (void *)m.index1_data().begin(), m.nnz()*sizeof(mf_size_type));
	std::memmove((void *)mc.index1_data().begin(), (void *)m.index2_data().begin(), m.nnz()*sizeof(mf_size_type));
	std::memmove((void *)mc.value_data().begin(), (void *)m.value_data().begin(), m.nnz()*sizeof(double));
	mc.set_filled(m.nnz());
	mc.sort();
}

}

#endif
