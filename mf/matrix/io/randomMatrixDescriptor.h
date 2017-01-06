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
/*
 * seedDescriptor.h
 *
 *  Created on: May 10, 2012
 *      Author: chteflio
 */




#ifndef MF_MATRIX_IO_RANDOMMATRIXDESCRIPTOR_H
#define MF_MATRIX_IO_RANDOMMATRIXDESCRIPTOR_H

#include <mf/matrix/distributed_matrix.h>
#include <util/random.h>

namespace mf {

/** Describes how to generate on the fly a distributed matrix */
struct RandomMatrixDescriptor {

	static RandomMatrixDescriptor create(mf_size_type size1, mf_size_type size2, mf_size_type chunks1, mf_size_type chunks2,
			mf_size_type nnz, mf_size_type nnzTest, mf_size_type rank, std::string values, std::string noise){
		BOOST_ASSERT(size1 % chunks1 == 0);
		BOOST_ASSERT(size2 % chunks2 == 0);


		RandomMatrixDescriptor result;
		rg::Random32 random = rg::Random32(123);

		result.size1 = size1;
		result.size2 = size2;
		result.chunks1 = chunks1;
		result.chunks2 = chunks2;
		result.nnz = nnz;
		result.nnzTest = nnzTest;
		result.values = values;
		result.noise = noise;
		result.rank = rank;

		result.nnzPerChunk.resize(chunks1,chunks2,false);
		result.nnzTestPerChunk.resize(chunks1,chunks2,false);
		result.seedsV.resize(chunks1,chunks2,false);
		result.seedsVtest.resize(chunks1,chunks2,false);

		//calculate seeds
		for (unsigned b1 = 0; b1 < chunks1; b1++){
			// seeds for Worig
			result.seedsWorig.push_back(random());
			for (unsigned b2 = 0; b2 < chunks2; b2++){
				// seeds for Horig
				if (b1==0){
					result.seedsHorig.push_back(random());
				}
				//seeds for V
				result.seedsV(b1,b2)=random();
				result.seedsVtest(b1,b2)=random();
			}
		}

		//calculate nnz per grid-cell
		if (nnz!=0 && nnzTest!=0){
			result.calculateNnzPerBlock(result.nnzPerChunk, result.nnz,result.chunks1, result.chunks2);
			result.calculateNnzPerBlock(result.nnzTestPerChunk, result.nnzTest,result.chunks1, result.chunks2);
		}

		//std::cout<<result.seedsV<<std::endl;
		return result;
	}

	void calculateNnzPerBlock(boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk, mf_size_type nnz,
			mf_size_type chunks1, mf_size_type chunks2);

	/** Stores the descriptor as XML file. */
	void load(const std::string &filename);

	/** Reads a descriptor from an XML file. */
	void save(const std::string &filename);

	// not sure if I need all these
	mf_size_type size1;
	mf_size_type size2;
	mf_size_type rank;
	mf_size_type chunks1;
	mf_size_type chunks2;
	// if nnz=0 I don't store V and Vtest, but only the initial W,H
	mf_size_type nnz;
	mf_size_type nnzTest;

	std::string values,noise;
	boost::numeric::ublas::matrix<mf_size_type> nnzPerChunk;
	boost::numeric::ublas::matrix<mf_size_type> nnzTestPerChunk;
	boost::numeric::ublas::matrix<unsigned> seedsV;
	boost::numeric::ublas::matrix<unsigned> seedsVtest;
	std::vector<unsigned> seedsWorig;
	std::vector<unsigned> seedsHorig;


};

} // namespace mf



#endif
