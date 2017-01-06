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
 * Some examples of loading and blocking single-file matrices into distributed memory.
 */

#include <iostream>

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/io.h>
#include <util/random.h>

#include <mpi2/mpi2.h>

#include <mf/mf.h>


using namespace std;
using namespace mpi2;
using namespace mf;
using namespace rg;
using namespace boost::numeric::ublas;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char* argv[]) {
	boost::mpi::communicator& world = mfInit(argc, argv);
	mfStart();

	if (world.rank() == 0) {
		mf_size_type size1 = 10;
		mf_size_type size2 = 8;
		mf_size_type nnz = 20;
		mf_size_type r = 2;
		///////////////////
		mf_size_type max;
		///////////////////
		// an empty distributed matrix
		SparseMatrix sTemp;
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Creating an empty distributed sparse matrix (2x2)");
		DistributedSparseMatrix z22("z22", size1, size2, 2, 2); // this creates only a descriptor
		z22.create();                                           // this creates the empty blocks
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", z22));
		LOG4CXX_INFO(logger, rg::paste("Nnz: ", mf::nnz(z22)));
		unblock(z22, sTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", sTemp));
		LOG4CXX_INFO(logger, rg::paste("Unblocked nnz: ", mf::nnz(sTemp)));

		// create some matrices to play with
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Generating some matrices: s, w, h");
		Random32 random;
		SparseMatrix s(size1, size2);
		DenseMatrix mDense(size1, size2), w(size1, r);
		DenseMatrixCM h(r, size2);
		generateRandom(s, nnz, random, boost::normal_distribution<>(0, 1));
		generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
		generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));

		// write them out
		LOG4CXX_INFO(logger, "Writing the matrices to disk");
		string fs = "/tmp/v";
		string fw = "/tmp/w";
		string fh = "/tmp/h";
		writeMatrix(fs, s, MM_COORD);
		writeMatrix(fw, w, MM_ARRAY);
		writeMatrix(fh, h, MM_ARRAY);

		// load as a distributed 1x1 sparse matrix
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Loading s as a distributed sparse matrix (1x1)");
		DistributedSparseMatrix s11 = loadMatrix<SparseMatrix>(
				"s11", 1, 1, true, fs, MM_COORD);
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", s11));
		unblock(s11, sTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", sTemp));
		LOG4CXX_INFO(logger, rg::paste("Unblocked nnz: ", mf::nnz(sTemp)));

		// load as a distributed 1x1 dense matrix
		DenseMatrix dTemp;
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Loading s as a distributed dense matrix (1x1)");
		DistributedDenseMatrix d11 = loadMatrix<DenseMatrix>(
				"d11", 1, 1, true, fs, MM_COORD);
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", d11));
		mDense.clear(); unblock(d11, dTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", dTemp));

		// load as a distributed 2x2 sparse matrix
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Loading s as a distributed sparse matrix (2x2)");
		DistributedSparseMatrix s22 = loadMatrix<SparseMatrix>(
				"s22", 2, 2, true, fs, MM_COORD);
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", s22));
		std::vector<mf_size_type> nnz1, nnz2;
		mf::nnz12(s22, nnz1, nnz2,max);///////////////////////////////////////////////
		LOG4CXX_INFO(logger, rg::paste("nnz1:", nnz1));
		LOG4CXX_INFO(logger, rg::paste("nnz2:", nnz2));
		unblock(s22, sTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", sTemp));
		LOG4CXX_INFO(logger, rg::paste("Unblocked nnz: ", mf::nnz(sTemp)));
		mf::nnz12(sTemp, nnz1, nnz2,max);///////////////////////////////////////
		LOG4CXX_INFO(logger, rg::paste("Unblocked nnz1:", nnz1));
		LOG4CXX_INFO(logger, rg::paste("Unblocked nnz2:", nnz2));

		// load a 2x1 dense matrix
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Loading w as a distributed dense matrix (2x1)");
		DistributedDenseMatrix w21 = loadMatrix<DenseMatrix>(
				"w21", 2, 1, true, fw, MM_ARRAY);
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", w21));
		mDense.clear(); unblock(w21, dTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", dTemp));

		// load a 1x2 dense matrix, partition by column
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Loading h as a distributed dense matrix (1x2, CM)");
		DistributedDenseMatrixCM h12 = loadMatrix<DenseMatrixCM>(
				"h12", 1, 2, false, fh, MM_ARRAY);
		LOG4CXX_INFO(logger, rg::paste("Descriptor:\n", h12));
		mDense.clear(); unblock(h12, dTemp);
		LOG4CXX_INFO(logger, rg::paste("Unblocked value: ", dTemp));

		// print the environments
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Printing the environments of each rank");
		lsAll();

		// some statistcs
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Comparing results of operations on matrices / distributed matrices");
		LOG4CXX_INFO(logger, "Nnz: " << mf::nnz(s) << " " << mf::nnz(s22) << " " << mf::nnz(s22, 2));
		LOG4CXX_INFO(logger, "Sum: " << sum(s) << " " << sum(s22) << " " << sum(s22, 2));
		LOG4CXX_INFO(logger, "Sum of squares: " << l2(s) << " " << l2(s22) << " " << l2(s22, 2));
		LOG4CXX_INFO(logger, "nzsl: " << nzsl(s, w, h) << " " << nzsl(s22, w21, h12) << " " << nzsl(s22, w21, h12, 2));
	}

	mfStop();
	mfFinalize();

	return 0;
}
