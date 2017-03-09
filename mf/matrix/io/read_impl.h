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
#include <mf/matrix/io/read.h>   // compiler hint

#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>

#include <boost/assert.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

#include <util/exception.h>
#include <util/io.h>

namespace mf {

namespace detail {

/**
 * @tparam Init function that initializes an output matrix (args: size1 size2 nnz)
 * @tparam CheckProcess function that checks whether an entry should be processed (args: i j)
 * @tparam Process function that adds an element to the output matrix (args: i j x)
 * @tparam Freeze function that freezes the matrix once read (no args)
 */
template<class Init, class CheckProcess, class Process, class Freeze>
void readMmCoord(const std::string& fname, Init init, CheckProcess checkProcess, Process process, Freeze freeze) {
	// open file
	std::ifstream in(fname.c_str());
	if (!in.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + fname);

	// check for correct file format
	std::string line;
	mf_size_type lineNumber = 0;
	if (!getline(in, line))
		RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);
	lineNumber++;
	if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix coordinate real general") == 0)
		RG_THROW(rg::IOException, std::string("Wrong matrix-market banner in file ") + fname
				+": " + line);

	// skip all comments
	while (getline(in, line) && line.at(0)=='%') {
		lineNumber++;
	}
	lineNumber++;
	if (line.at(0)=='%')
		RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);


	// read dimension line
	mf_size_type size1, size2, nnz;
	char junk[line.size()+1]; junk[0]=0;
	if (sscanf(line.c_str(), "%ld %ld %ld%[^\n]", &size1, &size2, &nnz, junk) < 3
			|| !boost::trim_left_copy(std::string(junk)).empty()) {
		RG_THROW(rg::IOException, std::string("Invalid matrix dimensions in file ") + fname + ": "+ line);
	}

	// initialize
	if (!init(size1, size2, nnz)) return;

	// read matrix
	for (mf_size_type p=0; p<nnz; p++) {
		if (!getline(in, line)) RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);
		lineNumber++;
		char *sAll = const_cast<char *>(line.c_str());
		char *sRemaining = sAll;
		char *sRemainingTemp = NULL;

		// get row/column
		mf_size_type i = strtoul(sRemaining, &sRemainingTemp, 10) - 1U;
		if (sRemaining==sRemainingTemp) RG_THROW(rg::IOException, rg::paste("Parse error at line ", lineNumber, " of ", fname, ": ", sAll));
		sRemaining = sRemainingTemp;
		mf_size_type j = strtoul(sRemaining, &sRemainingTemp, 10) - 1U;
		if (sRemaining==sRemainingTemp) RG_THROW(rg::IOException, rg::paste("Parse error at line ", lineNumber, " of ", fname, ": ", sAll));
		sRemaining = sRemainingTemp;

		// process value, if needed
		if ( checkProcess(i, j) ) {
			process(i, j, strtod(sRemaining, &sRemainingTemp));
			if (sRemaining==sRemainingTemp || !boost::trim_left_copy(std::string(sRemainingTemp)).empty() ) {
				RG_THROW(rg::IOException, rg::paste("Parse error at line ", lineNumber, " of ", fname, ": ", sAll));
			}
		}
	}

	// check that the rest of the file is emty
	while (getline(in, line)) {
		lineNumber++;
		if (!boost::trim_left_copy(std::string(line)).empty()) {
			RG_THROW(rg::IOException, rg::paste("Unexpected input at at line ", lineNumber, " of ", fname, ": ", line));
		}
	}

	freeze();
}

template<class M>
inline bool mmInit(M& m, bool read, mf_size_type size1, mf_size_type size2, mf_size_type nnz) {
	m.resize(size1, size2, false);
	m.clear();
	return read;
}

template<typename M>
inline bool mmCheckProcess(M& m, mf_size_type i, mf_size_type j) {
	return true;
}

template<typename M>
inline void mmProcess(M& m, mf_size_type i, mf_size_type j, typename M::value_type x) {
	m(i,j) = x;
}

template<typename M>
inline void mmFreeze(M& m) {
}

template<class L, std::size_t IB, class IA, class TA>
inline bool mmInitSparse(boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m,
		bool read, mf_size_type size1, mf_size_type size2, mf_size_type nnz) {
	m.resize(size1, size2, false);
	m.clear();
	m.reserve(nnz);
	return read;
}

template<class L, std::size_t IB, class IA, class TA>
inline bool mmCheckProcessSparse(boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m,
		mf_size_type i, mf_size_type j) {
	return true;
}

template<class L, std::size_t IB, class IA, class TA>
inline void mmProcessSparse(boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m,
		mf_size_type i, mf_size_type j,
		typename boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>::value_type x) {
	m.append_element(i, j, x);
}

template<class L, std::size_t IB, class IA, class TA>
inline void mmFreezeSparse(boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m) {
	m.sort();
}

template<class M>
void readMmCoord(const std::string& fname, M& m) {
	readMmCoord(
			fname,
			boost::bind(mmInit<M>, boost::ref(m), true, _1, _2, _3),
			boost::bind(mmCheckProcess<M>, boost::ref(m), _1, _2),
			boost::bind(mmProcess<M>, boost::ref(m), _1, _2, _3),
			boost::bind(mmFreeze<M>, boost::ref(m))
			);
}

template<class L, std::size_t IB, class IA, class TA>
void readMmCoord(const std::string& fname, boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m) {
	readMmCoord(
			fname,
			boost::bind(mmInitSparse<L,IB,IA,TA>, boost::ref(m), true, _1, _2, _3),
			boost::bind(mmCheckProcessSparse<L,IB,IA,TA>, boost::ref(m), _1, _2),
			boost::bind(mmProcessSparse<L,IB,IA,TA>, boost::ref(m), _1, _2, _3),
			boost::bind(mmFreezeSparse<L,IB,IA,TA>, boost::ref(m))
			);
}

/**
 * @tparam Init function that initializes an output matrix (args: size1 size2 nnz)
 * @tparam CheckProcess function that checks whether an entry should be processed (args: i j)
 * @tparam Process function that adds an element to the output matrix (args: i j x)
 * @tparam Freeze function that freezes the matrix once read (no args)
 */
template<class Init, class CheckProcess, class Process, class Freeze>
void readMmArray(const std::string& fname, Init init, CheckProcess checkProcess, Process process, Freeze freeze) {
	// open file
	std::ifstream in(fname.c_str());
	if (!in.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + fname);

	// check for correct file format
	std::string line;
	mf_size_type lineNumber = 0;
	if (!getline(in, line))
		RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);
	lineNumber++;
	if (!boost::trim_right_copy(line).compare("%%MatrixMarket matrix array real general") == 0)
		RG_THROW(rg::IOException, std::string("Wrong matrix-market banner in file ") + fname
				+": " + line);

	// skip all comments
	while (getline(in, line) && line.at(0)=='%') lineNumber++;
	lineNumber++;
	if (line.at(0)=='%')
		RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);

	// read dimension line
	mf_size_type size1, size2;
	char junk[line.size()+1]; junk[0]=0;
	if (sscanf(line.c_str(), "%ld %ld%[^\n]", &size1, &size2, junk) < 2
			|| !boost::trim_left_copy(std::string(junk)).empty()) {
		RG_THROW(rg::IOException, std::string("Invalid matrix dimensions in file ") + fname + ": "+ line);
	}

	// resize matrix
	init(size1, size2, size1*size2);

	// read matrix
	for (mf_size_type j=0; j<size2; j++) {
		for (mf_size_type i=0; i<size1; i++) {
			if (!getline(in, line)) RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + fname);
			lineNumber++;

			// process value, if needed
			if ( checkProcess(i,j) ) {
				char* sAll = const_cast<char *>(line.c_str());
				char* sRemaining = NULL;
				double x = strtod(line.c_str(), &sRemaining);
				if (sAll == sRemaining || !boost::trim_left_copy(std::string(sRemaining)).empty() ) {
					RG_THROW(rg::IOException, rg::paste("Parse error at line ", lineNumber, " of ", fname, ": ", sAll));
				}
				process(i, j, x);
			}
		}
	}

	// check that the rest of the file is emty
	while (getline(in, line)) {
		lineNumber++;
		if (!boost::trim_left_copy(std::string(line)).empty()) {
			RG_THROW(rg::IOException, rg::paste("Unexpected input at at line ", lineNumber, " of ", fname, ": ", line));
		}
	}

	freeze();
}

template<class M>
void readMmArray(const std::string& fname, M& m) {
	readMmArray(
			fname,
			boost::bind(mmInit<M>, boost::ref(m), true, _1, _2, _3),
			boost::bind(mmCheckProcess<M>, boost::ref(m), _1, _2),
			boost::bind(mmProcess<M>, boost::ref(m), _1, _2, _3),
			boost::bind(mmFreeze<M>, boost::ref(m))
			);
}

template<class L, std::size_t IB, class IA, class TA>
void readMmArray(const std::string& fname, boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>& m) {
	readMmArray(
			fname,
			boost::bind(mmInitSparse<L,IB,IA,TA>, boost::ref(m), true, _1, _2, _3),
			boost::bind(mmCheckProcessSparse<L,IB,IA,TA>, boost::ref(m), _1, _2),
			boost::bind(mmProcessSparse<L,IB,IA,TA>, boost::ref(m), _1, _2, _3),
			boost::bind(mmFreezeSparse<L,IB,IA,TA>, boost::ref(m))
			);
}

template<class Matrix>
void readBoostText(const std::string& fname, Matrix& M) {
	std::ifstream in(fname.c_str());
	boost::archive::text_iarchive ia(in);
	ia >> M;
}

template<class Matrix>
void readBoostBin(const std::string& fname, Matrix& M) {
	std::ifstream in(fname.c_str());
	boost::archive::binary_iarchive ia(in);
	ia >> M;
}

} // namespace detail

template<typename M>
void readMatrix(const std::string& fname, M& m, MatrixFileFormat format) {
	if (format == AUTOMATIC) {
		format = getMatrixFormat(fname);
	}
	switch (format) {
	case AUTOMATIC: // handled above
	case MM_ARRAY:
		detail::readMmArray(fname, m);
		break;
	case MM_COORD:
		detail::readMmCoord(fname, m);
		break;
	case BOOST_SPARSE_BIN:
	case BOOST_DENSE_BIN:
		detail::readBoostBin(fname, m);
		break;
	case BOOST_SPARSE_TEXT:
	case BOOST_DENSE_TEXT:
		detail::readBoostText(fname, m);
		break;
	default:
		RG_THROW(rg::InvalidArgumentException, "invalid matrix format");
	}
}

namespace detail {
	template<typename M, bool Sparse = false>
	struct InitProcess {
		inline static bool init(M& m, bool read, mf_size_type size1, mf_size_type size2, mf_size_type nnz) {
			return mmInit(m, read, size1, size2, nnz);
		}

		inline static bool checkProcess(M& m, mf_size_type i, mf_size_type j) {
			return mmCheckProcess(m, i, j);
		}

		inline static void process(M& m, mf_size_type i, mf_size_type j, typename M::value_type x) {
			mmProcess(m, i, j, x);
		}

		inline static void freeze(M& m) {
			mmFreeze(m);
		}
	};

	template<typename M>
	struct InitProcess<M, true> {
		inline static bool init(M& m, bool read, mf_size_type size1, mf_size_type size2, mf_size_type nnz) {
			return mmInitSparse(m, read, size1, size2, nnz);
		}

		inline static bool checkProcess(M& m, mf_size_type i, mf_size_type j) {
			return mmCheckProcessSparse(m, i, j);
		}

		inline static void process(M& m, mf_size_type i, mf_size_type j, typename M::value_type x) {
			mmProcessSparse(m, i, j, x);
		}

		inline static void freeze(M& m) {
			mmFreezeSparse(m);
		}
	};

	/** Main worker class for efficient blocking of matrices stored in matrix-market format.
	 * Builds an index that allows to quickly find the block to which an entry read from the input
	 * belongs to (if any). */
	template<typename M, bool SparseIn, bool SparseOut>
	class ReadMatrixBlocksMm {
	public:
		ReadMatrixBlocksMm(
				mf_size_type blocks1,
				mf_size_type blocks2,
				const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
				std::vector<mf_size_type>& blockOffsets1,
				std::vector<mf_size_type>& blockOffsets2,
				mf_size_type& size1,
				mf_size_type& size2,
				std::vector<M*>& blocks)
		: blocks1(blocks1), blocks2(blocks2), sortedBlockList(sortedBlockList),
		  blockOffsets1(blockOffsets1), blockOffsets2(blockOffsets2),
		  size1(size1), size2(size2), blocks(blocks)
		{
		}

		/** Creates an empty matrix for each block and computes the block index */
		inline bool init(bool read, mf_size_type size1, mf_size_type size2, mf_size_type nnz) {
			this->size1 = size1;
			this->size2 = size2;

			// create block offsets (if not given)
			if (blockOffsets1.empty()) computeDefaultBlockOffsets(size1, blocks1, blockOffsets1);
			if (blockOffsets2.empty()) computeDefaultBlockOffsets(size2, blocks2, blockOffsets2);

			// create block index
			createBlockIndex();

			// create result vector
			blocks.clear();
			for (unsigned i=0; i<sortedBlockList.size(); i++) {
				mf_size_type b1 = sortedBlockList[i].first;
				mf_size_type b2 = sortedBlockList[i].second;
				mf_size_type bsize1 = blockSize(b1, size1, blockOffsets1);
				mf_size_type bsize2 = blockSize(b2, size2, blockOffsets2);
				M* m = new M(bsize1, bsize2);
				mf_size_type bnnz = SparseIn ? (double)nnz/(blocks1*blocks2)*1.1 : bsize1*bsize2; // overallocate 10% of average block size
				InitProcess<M, SparseOut>::init(*m, read, bsize1, bsize2, bnnz);
				blocks.push_back(m);
			}

			return read;
		}

		inline bool checkProcess(mf_size_type i, mf_size_type j) {
			int b = blockOf(i, j, blockIndex1, blockIndex2);
			return b>=0;
		}

		/** Puts the matrix entry into the right block (if any) */
		inline void process(mf_size_type i, mf_size_type j, typename M::value_type x) {
			int b = blockOf(i, j, blockIndex1, blockIndex2);
			BOOST_ASSERT( b>=0 );
			mf_size_type b1 = sortedBlockList[b].first;
			mf_size_type b2 = sortedBlockList[b].second;
			InitProcess<M, SparseOut>::process(*blocks[b], i-blockOffsets1[b1], j-blockOffsets2[b2], x);
		}

		inline void freeze() {
			BOOST_FOREACH(M* block, blocks) {
				InitProcess<M, SparseOut>::freeze(*block);
			}
		}

	private:
		/** Creates the block index. The row index (blockIndex1) contains the position
		 * of the first block on that row (or -1 if none). The columnindex (blockIndex2)
		 * contains the offset to the block of the column (or -1 if none). When both
		 * indexes are nonnegative, the sum blockIndex1[i]+blockIndex2[j] points to the
		 * block in the blocklist to which (i,j) belongs.
		 */
		inline void createBlockIndex() {
			// initialize
			blockIndex1.clear();
			blockIndex1.resize(size1, -1);
			blockIndex2.clear();
			blockIndex2.resize(size2, -1);

			// go
			int currentBlock1 = -1, listOffset1, listOffset2;
			typedef std::pair<mf_size_type, mf_size_type> Block;
			for (unsigned i=0; i<sortedBlockList.size(); i++) {
				mf_size_type b1 = sortedBlockList[i].first;
				mf_size_type b2 = sortedBlockList[i].second;
				if ((int)b1 != currentBlock1) {
					listOffset1 = i;
					listOffset2 = 0;
					currentBlock1 = b1;
				} else {
					listOffset2++;
				}

				mf_size_type low1 = blockOffsets1[b1];
				mf_size_type high1 = b1+1 < blockOffsets1.size() ? blockOffsets1[b1+1] : size1;
				std::fill(blockIndex1.begin()+low1, blockIndex1.begin()+high1, listOffset1);


				mf_size_type low2 = blockOffsets2[b2];
				mf_size_type high2 = b2+1 < blockOffsets2.size() ? blockOffsets2[b2+1] : size2;
				std::fill(blockIndex2.begin()+low2, blockIndex2.begin()+high2, listOffset2);
			}
		}

		inline int blockOf(mf_size_type i, mf_size_type j,
				const std::vector<int>& blockIndex1, const std::vector<int>& blockIndex2) {
			if (blockIndex1[i] < 0 || blockIndex2[j] < 0) {
				return -1;
			} else {
				return blockIndex1[i] + blockIndex2[j];
			}
		}

		mf_size_type blocks1, blocks2;
		const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList;
		std::vector<mf_size_type>& blockOffsets1;
		std::vector<mf_size_type>& blockOffsets2;
		mf_size_type& size1;
		mf_size_type& size2;
		std::vector<M*>& blocks;
		std::vector<int> blockIndex1, blockIndex2;
	};


	/** Selects the method to read the input depending on whether the input file is
	 * in one of the matrix market format and whether the output file is sparse. Falls
	 * back to slow default method for non matrix market formats. */
	template<typename M, bool SparseOut>
	void readMatrixBlocks(const std::string& fname,
			mf_size_type blocks1, mf_size_type blocks2,
			const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
			std::vector<mf_size_type>& blockOffsets1, std::vector<mf_size_type>& blockOffsets2,
			mf_size_type& size1, mf_size_type& size2, std::vector<M*>& blocks,
			MatrixFileFormat format = AUTOMATIC) {
		typedef std::pair<mf_size_type, mf_size_type> Block;

		if (format == AUTOMATIC) {
			format = getMatrixFormat(fname);
		}

		switch (format) {
		case MM_COORD:
		{
			if (sortedBlockList.size() > 0) {
				if (!SparseOut) {
					LOG4CXX_WARN(detail::logger, "Input file '" << fname
							<< "' is in MM_COORD format but read into a dense matrix");
				}
				std::stringstream ss;
				ss << "Constructing blocks ";
				BOOST_FOREACH(Block b, sortedBlockList) {
					ss << "(" << b.first << "," << b.second << ")" << " ";
				}
				ss << " of '" << fname << "'";
				LOG4CXX_INFO(detail::logger, ss.str());
			}

			ReadMatrixBlocksMm<M, true, SparseOut> reader(
					blocks1, blocks2, sortedBlockList,
					blockOffsets1, blockOffsets2,
					size1, size2, blocks);

			readMmCoord(
					fname,
					boost::bind(&ReadMatrixBlocksMm<M, true, SparseOut>::init,
							boost::ref(reader),
							!sortedBlockList.empty(), _1, _2, _3),
					boost::bind(&ReadMatrixBlocksMm<M, true, SparseOut>::checkProcess,
							boost::ref(reader),
							_1, _2),
					boost::bind(&ReadMatrixBlocksMm<M, true, SparseOut>::process,
							boost::ref(reader),
					_1, _2, _3),
					boost::bind(&ReadMatrixBlocksMm<M, true, SparseOut>::freeze,
							boost::ref(reader))
			);
		}
		break;
		case MM_ARRAY:
		{
			if (sortedBlockList.size() > 0) {
				if (SparseOut) {
					LOG4CXX_WARN(detail::logger, "Input file '" << fname
							<< "' is in MM_ARRAY format but read into a sparse matrix");
				}
				std::stringstream ss;
				ss << "Constructing blocks ";
				BOOST_FOREACH(Block b, sortedBlockList)
				ss << "(" << b.first << "," << b.second << ")" << " ";
				ss << " of '" << fname << "'";
				LOG4CXX_INFO(detail::logger, ss.str());
			}

			ReadMatrixBlocksMm<M, false, SparseOut> reader(
					blocks1, blocks2, sortedBlockList,
					blockOffsets1, blockOffsets2,
					size1, size2, blocks);

			readMmArray(
					fname,
					boost::bind(&ReadMatrixBlocksMm<M, false, SparseOut>::init,
							boost::ref(reader),
							!sortedBlockList.empty(), _1, _2, _3),
					boost::bind(&ReadMatrixBlocksMm<M, false, SparseOut>::checkProcess,
							boost::ref(reader),
							_1, _2),
					boost::bind(&ReadMatrixBlocksMm<M, false, SparseOut>::process,
							boost::ref(reader),
							_1, _2, _3),
					boost::bind(&ReadMatrixBlocksMm<M, false, SparseOut>::freeze,
						boost::ref(reader))
			);
		}
		break;
		default:
			LOG4CXX_WARN(detail::logger, "Input matrix '" << fname
					<< "' is not in MM format: readMatrixBlocks() will be memory intensive");
			if (SparseOut) {
				LOG4CXX_WARN(detail::logger, "Input matrix '" << fname
						<< "' is not in MM format and output matrix is sparse: "
						<< " readMatrixBlocks() will be slow");
			}

			// read the entire matrix (wastes memory)
			M m;
			LOG4CXX_INFO(detail::logger, "Reading " << fname << "...");
			readMatrix(fname, m, format);
			size1 = m.size1();
			size2 = m.size2();
			if (blockOffsets1.empty()) computeDefaultBlockOffsets(size1, blocks1, blockOffsets1);
			if (blockOffsets2.empty()) computeDefaultBlockOffsets(size2, blocks2, blockOffsets2);

			// block it (wastes time for sparse matrices)
			blocks.clear();
			blocks.reserve(sortedBlockList.size());
			typedef std::pair<mf_size_type, mf_size_type> Block;
			BOOST_FOREACH(Block b, sortedBlockList) {
				mf_size_type b1 = b.first;
				mf_size_type b2 = b.second;
				LOG4CXX_INFO(detail::logger, "Blocking (" << b1 << "," << b2 << ")");
				mf_size_type rowLow = blockOffsets1[b1];
				mf_size_type rowHigh = b1+1 < blockOffsets1.size() ? blockOffsets1[b1+1] : m.size1();
				mf_size_type colLow = blockOffsets2[b2];
				mf_size_type colHigh = b2+1 < blockOffsets2.size() ? blockOffsets2[b2+1] : m.size2();
				M *block = new M();
				projectSubrange(m, *block, rowLow, rowHigh, colLow, colHigh);
				blocks.push_back(block);
			};
			break;
		}
	}
}

template<class M>
void readMatrixBlocks(
		const std::string& fname,
		mf_size_type blocks1, mf_size_type blocks2,
		const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
		std::vector<mf_size_type>& blockOffsets1, std::vector<mf_size_type>& blockOffsets2,
		mf_size_type& size1, mf_size_type& size2,
		std::vector<M*>& blocks,
		MatrixFileFormat format) {
	detail::readMatrixBlocks<M, false>(fname, blocks1, blocks2, sortedBlockList,
			blockOffsets1, blockOffsets2, size1, size2, blocks, format);
}

template<class L, std::size_t IB, class IA, class TA>
void readMatrixBlocks(
		const std::string& fname,
		mf_size_type blocks1, mf_size_type blocks2,
		const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
		std::vector<mf_size_type>& blockOffsets1, std::vector<mf_size_type>& blockOffsets2,
		mf_size_type& size1, mf_size_type& size2,
		std::vector<boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>*>& blocks,
		MatrixFileFormat format) {
	typedef boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA> M;
	detail::readMatrixBlocks<M, true>(fname, blocks1, blocks2, sortedBlockList,
			blockOffsets1, blockOffsets2, size1, size2, blocks, format);
	typedef boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA> M;
}

}

