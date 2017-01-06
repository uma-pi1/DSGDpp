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
 *
 * Defines standard types used by the mf library.
 */

#ifndef MF_TYPES_H
#define MF_TYPES_H

#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <util/exception.h>

#include <mpi2/mpi2.h>

namespace mf {
// common matrix types
typedef boost::numeric::ublas::coordinate_matrix<double, boost::numeric::ublas::row_major>
	SparseMatrix;
typedef boost::numeric::ublas::coordinate_matrix<double, boost::numeric::ublas::column_major>
	SparseMatrixCM;
typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major>
	DenseMatrix;
typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>
	DenseMatrixCM;

typedef SparseMatrix::size_type mf_size_type;
}

// register matrix types
MPI2_TYPE_TRAITS(mf::SparseMatrix);
MPI2_TYPE_TRAITS(mf::SparseMatrixCM);
MPI2_TYPE_TRAITS(mf::DenseMatrix);
MPI2_TYPE_TRAITS(mf::DenseMatrixCM);

namespace mf {

/** A list of sparse matrix types to be registered to the mf library. */
typedef
		mpi2::Cons<SparseMatrix,
		mpi2::Cons<SparseMatrixCM
		> >
SparseMatrixTypes;

/** A list of dense matrix types to be registered to the mf library. */
typedef
		mpi2::Cons<DenseMatrix,
		mpi2::Cons<DenseMatrixCM
		> >
DenseMatrixTypes;

/** A list of all matrix types. */
typedef mpi2::Concat<SparseMatrixTypes, DenseMatrixTypes> MatrixTypes;

/** A list of all built-in types relevant to the mf package. */
typedef
		MatrixTypes
MfBuiltinTypes;

}

#endif
