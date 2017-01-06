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
 * generateDistributedMatrix_impl.h
 *
 *  Created on: May 15, 2012
 *      Author: chteflio
 *
 *  Comments: We had problems with mpi2 when creating very large matrices on 16 nodes.
 *  if you run into such issues, go to generateDataMatrix function and for the line:
 *  runTaskOnBlocks<GenerateRandomDataMatrixTask<M> >(m, arg, std::min(tasksPerRank, blocks1PerNode));
 *  provide a different implementation for runTaskOnBlocks where you replace
 *
 *	mpi2::sendEach(channels, args);
 *
 *	with
 *
 *	for (int i=0; i<channels.size(); i++){
 *		channels[i].send(args[i]);
 *	}
 */

#include <mf/matrix/io/generateDistributedMatrix.h>   // compiler hint
#include <mf/matrix/distribute.h> // IDE hint
#include <tools/parse.h>
#include <mf/matrix/io/loadDistributedMatrix.h>
#include <util/evaluation.h>

namespace mf { namespace detail {

template<typename M>
struct GenerateArgs {
	//to be called for DataMatrix
	GenerateArgs(M* v, rg::Random32& random, mf_size_type start1, mf_size_type end1, mf_size_type start2, mf_size_type end2,
			mf_size_type nnz, mf_size_type nnzStartPoint, const std::string& noise)
	:v(v), random(random),
	 start1(start1), end1(end1), start2(start2), end2(end2), nnz(nnz),
	 nnzStartPoint(nnzStartPoint), noise(noise){};

	//to be called for FactorMatrix
	GenerateArgs(M* v, rg::Random32 random,
			mf_size_type start1, mf_size_type end1, mf_size_type start2, mf_size_type end2,
			const std::string values)
	:v(v), random(random), start1(start1), end1(end1), start2(start2), end2(end2), values(values){};

	M* v;
	mf_size_type start1, end1, start2, end2, nnz, nnzStartPoint;
	std::string values, noise;
	rg::Random32 random;
};

template<typename M>
struct GenerateValues {
	GenerateValues(GenerateArgs<M>& args) : args(args) {	};

	static void run(GenerateArgs<M>& args) {
		GenerateValues<M> f(args);
		parse::parseDistribution("values", args.values, f);
	}
	template<typename Dist>
	void operator()(Dist dist) {
		generateRandom(*args.v, args.random, dist, args.start1, args.end1, args.start2, args.end2);
	};
	GenerateArgs<M>& args;
};
template<typename M>
struct GenerateNoise {
	GenerateNoise(GenerateArgs<M>& args) : args(args) {	};

	static void run(GenerateArgs<M>& args) {
		GenerateNoise<M> f(args);
		parse::parseDistribution("noise", args.noise, f);
	}
	template<typename Dist>
	void operator()(Dist dist) {
		addRandom(*args.v, args.random, dist, args.nnz, args.nnzStartPoint, args.start1, args.end1, args.start2, args.end2);
	};
	GenerateArgs<M>& args;
};

/* m is full matrix, only parts of it generated; m must have correct size */
template<typename M>
void generateFactor(M* m, const std::string& dist, const std::vector<unsigned>& seeds,
		bool rowBlocks, mf_size_type startChunk, mf_size_type endChunk) {
	mf_size_type size = rowBlocks ? m->size1() : m->size2();
	mf_size_type chunkSize = size / seeds.size();

	std::vector<mf_size_type> chunkOffsets;
	computeDefaultBlockOffsets(size, seeds.size(), chunkOffsets);

	// start and end refer to actual rows of the matrix
	for (int b=startChunk; b<endChunk; b++) {
		//mf_size_type start = b*chunkSize;
		//mf_size_type end = start + chunkSize;
		mf_size_type start = chunkOffsets[b];
		mf_size_type end = (b+1 < chunkOffsets.size() ? chunkOffsets[b+1] : size);

		rg::Random32 random = rg::Random32(seeds[b]);

		if (rowBlocks){
			GenerateArgs<M> args(m,random,start, end, 0, m->size2(),dist);
			GenerateValues<M>::run(args);
			//generateRandom(*m, random, boost::uniform_real<>(-0.5, 0.5), start, end, 0, m->size2());

		}else{
			GenerateArgs<M> args(m,random, 0, m->size1(),start, end,dist);
			GenerateValues<M>::run(args);
			//generateRandom(*m, random, boost::uniform_real<>(-0.5, 0.5), 0, m->size1(), start, end);
		}
	}
}
/* m is full matrix, only parts of it generated; m must have correct size, h might be larger than m
 * hOffset is the column of h that corresponds to the 1st column of m
 * */
template<typename M>
mf_size_type generateDataMatrix(M* v, const DenseMatrix* w, const DenseMatrixCM* h,
		const boost::numeric::ublas::matrix<unsigned>& seeds,
		const boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk,
		mf_size_type startChunk1, mf_size_type endChunk1, mf_size_type startChunk2, mf_size_type endChunk2,
		const boost::numeric::ublas::matrix<mf_size_type>& nnzStartPoints,
		mf_size_type hOffset, const std::string& values, const std::string& noise){

	mf_size_type size1 = v->size1();
	mf_size_type size2 = v->size2();
	mf_size_type blockSize1 = size1 / seeds.size1();
	mf_size_type blockSize2 = size2 / seeds.size2();

	std::vector<mf_size_type> chunkOffsets1, chunkOffsets2;
	computeDefaultBlockOffsets(size1, seeds.size1(), chunkOffsets1);
	computeDefaultBlockOffsets(size2, seeds.size2(), chunkOffsets2);

	mf_size_type nnzAdded = 0;
	// start and end refer to actual rows of the matrix
	for (mf_size_type b1=startChunk1; b1<endChunk1; b1++){
		for (mf_size_type b2=startChunk2; b2<endChunk2; b2++){

			//mf_size_type start1 = b1*blockSize1;
			//mf_size_type end1 = start1 + blockSize1;
			//mf_size_type start2 = b2*blockSize2;
			//mf_size_type end2 = start2 + blockSize2;

			mf_size_type start1 = chunkOffsets1[b1];
			mf_size_type end1 = (b1+1 < chunkOffsets1.size() ? chunkOffsets1[b1+1] : size1);
			mf_size_type start2 = chunkOffsets2[b2];
			mf_size_type end2 = (b2+1 < chunkOffsets2.size() ? chunkOffsets2[b2+1] : size2);

			nnzAdded += nnzPerChunk(b1,b2);

			rg::Random32 random = rg::Random32(seeds(b1,b2));

			generateRandom((*v), nnzPerChunk(b1,b2), (*w), (*h), random, nnzStartPoints(b1,b2), hOffset, start1, end1, start2, end2);
			v->set_filled(nnzAdded);

			if(noise != ""){//not a test
				GenerateArgs<M> args(v, random, start1, end1, start2, end2, nnzPerChunk(b1,b2), nnzStartPoints(b1,b2), noise);
				GenerateNoise<M>::run(args);
				//addRandom((*v), randomV, boost::normal_distribution<>(0, 0.1),nnzPerBlock(b1,b2),nnzStartPoints(b1,b2), start1,end1,start2,end2);
			}
		}
	}

	return nnzAdded;
}

/* Each thread operates on a different (but whole) row of chunks
 * This implies that the grid should be fine-grained enough on the chunks1 dimension
 * */
template<typename M>
struct ParallelGenerateDataMatrixTask {
	static const std::string id() { return std::string("__mf/.../ParallelGenerateDataMatrixTask_")
	+ mpi2::TypeTraits<M>::name(); }
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		// receive pointers
		mpi2::PointerIntType pv, pw, ph, pseeds, psplit, pnnzPerChunk, pnnzStartPoints;
		mf_size_type hOffset;

		std::string values, noise;
		mf_size_type nnzAdded = 0;

		ch.recv(*mpi2::unmarshal(pv, pw, ph));
		M* v = mpi2::intToPointer<M>(pv);
		DenseMatrix* w = mpi2::intToPointer<DenseMatrix>(pw);
		DenseMatrixCM* h = mpi2::intToPointer<DenseMatrixCM>(ph);

		ch.recv(*mpi2::unmarshal(pseeds, pnnzPerChunk, psplit));
		boost::numeric::ublas::matrix<unsigned>& seeds = *mpi2::intToPointer<boost::numeric::ublas::matrix<unsigned> >(pseeds);
		boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk = *mpi2::intToPointer<boost::numeric::ublas::matrix<mf_size_type> >(pnnzPerChunk);
		std::vector<mf_size_type>& split = *mpi2::intToPointer<std::vector<mf_size_type> >(psplit);

		ch.recv(*mpi2::unmarshal(pnnzStartPoints, hOffset));
		boost::numeric::ublas::matrix<mf_size_type>& nnzStartPoints = *mpi2::intToPointer<boost::numeric::ublas::matrix<mf_size_type> >(pnnzStartPoints);

		ch.recv(*mpi2::unmarshal(values, noise));
		// run
		int p = info.groupId();
		// I assume that each thread operates on a row of chunks
		nnzAdded=generateDataMatrix<M>(v, w, h, seeds, nnzPerChunk,
				split[p], split[p+1], 0, seeds.size2(),
				nnzStartPoints,	hOffset, values, noise);

		// signal we are done
		ch.economicSend(nnzAdded, mpi2::TaskManager::getInstance().pollDelay());
	}
};

template<typename M>
void generateDataMatrix(M* v, const DenseMatrix* w, const DenseMatrixCM* h,
		const boost::numeric::ublas::matrix<unsigned>& seeds,
		const boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk,
		mf_size_type startChunk1, mf_size_type endChunk1, mf_size_type startChunk2, mf_size_type endChunk2,
		const boost::numeric::ublas::matrix<mf_size_type>& nnzStartPoints,
		mf_size_type hOffset, const std::string& values, const std::string& noise, int threads) {
	BOOST_ASSERT( threads > 0 );
	mf_size_type nnz = 0;

	mf_size_type nnzCalc = nnzStartPoints(nnzStartPoints.size1()-1, nnzStartPoints.size2()-1)+
			nnzPerChunk(nnzPerChunk.size1()-1, nnzPerChunk.size2()-1);

	v->reserve(nnzCalc);

	if (threads == 1) {
		nnz = generateDataMatrix(v, w, h,seeds, nnzPerChunk,
				startChunk1, endChunk1, startChunk2, endChunk2,	nnzStartPoints,	hOffset,
				values, noise);
	} else {

		typedef detail::ParallelGenerateDataMatrixTask<M> Task;
		std::vector<mf_size_type> split = mpi2::split(endChunk1-startChunk1, threads);
		for (int i=0; i<threads; i++) split[i] += startChunk1;

		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;

		tm.spawn<Task>(tm.world().rank(), threads, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(v), mpi2::pointerToInt(w),
				mpi2::pointerToInt(h)));
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&seeds), mpi2::pointerToInt(&nnzPerChunk),
				mpi2::pointerToInt(&split)));
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&nnzStartPoints), hOffset));


		mpi2::sendAll(channels, mpi2::marshal(values, noise));
		std::vector<mf_size_type> nnzVec(threads);
		mpi2::economicRecvAll(channels, nnzVec,10* tm.pollDelay());

		for (int i=0; i<nnzVec.size(); i++){
			nnz+=nnzVec[i];
		}

	}
	v->set_filled(nnz);
	v->sort();
}
/* Each thread operates on a different row-chunk
 * This implies that the grid should be fine-grained enough on the chunks1 dimension
 * */
template<typename M>
struct ParallelGenerateFactorTask {
	static const std::string id() { return std::string("__mf/.../ParallelGenerateFactorTask_")
	+ mpi2::TypeTraits<M>::name(); }
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		// receive pointers
		mpi2::PointerIntType pm, pseeds, psplit;
		ch.recv(*mpi2::unmarshal(pm, pseeds, psplit));
		M* m = mpi2::intToPointer<M>(pm);
		std::vector<unsigned>& seeds = *mpi2::intToPointer<std::vector<unsigned> >(pseeds);
		std::vector<mf_size_type>& split = *mpi2::intToPointer<std::vector<mf_size_type> >(psplit);

		// receive other arguments
		std::string dist;
		bool rowBlocks;
		ch.recv(*mpi2::unmarshal(dist, rowBlocks));

		// run
		int p = info.groupId();
		generateFactor<M>(m, dist, seeds, rowBlocks, split[p], split[p+1]);

		// signal we are done
		ch.economicSend(mpi2::TaskManager::getInstance().pollDelay());
	}
};

template<typename M>
void generateFactor(M* m, const std::string& dist, const std::vector<unsigned>& seeds,
		bool rowBlocks, mf_size_type startChunk, mf_size_type endChunk, int threads) {

	BOOST_ASSERT( threads > 0 );
	if (threads == 1) {
		return generateFactor(m, dist, seeds, rowBlocks, startChunk, endChunk);

	} else {
		typedef detail::ParallelGenerateFactorTask<M> Task;
		std::vector<mf_size_type> split = mpi2::split(endChunk-startChunk, threads);
		for (int i=0; i<threads; i++) split[i] += startChunk;
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;
		tm.spawn<Task>(tm.world().rank(), threads, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(m), mpi2::pointerToInt(&seeds),
				mpi2::pointerToInt(&split)));
		mpi2::sendAll(channels, mpi2::marshal(dist, rowBlocks));
		mpi2::economicRecvAll(channels, tm.pollDelay());
	}
}


struct GenerateRandomFactorsTaskArg {
public:
	GenerateRandomFactorsTaskArg() {};

	// to be called when W or H is to be generated
	GenerateRandomFactorsTaskArg(
			const std::string& values, const std::vector<unsigned>& seeds, bool rowBlocks,
			int chunksPerBlock, int threadsPerTask)
	: values(values), seeds(seeds), rowBlocks(rowBlocks),
	  chunksPerBlock(chunksPerBlock), threadsPerTask(threadsPerTask)
	{}

	std::string values;
	std::vector<unsigned> seeds;
	bool rowBlocks;
	int threadsPerTask;
	int chunksPerBlock;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & values;
		ar & seeds;
		ar & rowBlocks;
		ar & threadsPerTask;
		ar & chunksPerBlock;
	}
};

struct GenerateRandomDataMatrixTaskArg {
public:
	GenerateRandomDataMatrixTaskArg() {};

	// to be called when W or H is to be generated
	GenerateRandomDataMatrixTaskArg(
			const std::string& values,const std::string& noise,
			const boost::numeric::ublas::matrix<unsigned>& seeds,
			const boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk,
			const std::vector<std::string>& wBlockNames,
			const std::vector<std::string>& hBlockNames,
			const std::vector<mf_size_type>& blockOffsets2,
			int chunksPerBlock1, int chunksPerBlock2,
			int threadsPerTask, bool forDapCM)
	: values(values),noise(noise), seeds(seeds), nnzPerChunk(nnzPerChunk),
	  wBlockNames(wBlockNames), hBlockNames(hBlockNames),blockOffsets2(blockOffsets2),
	  chunksPerBlock1(chunksPerBlock1),chunksPerBlock2(chunksPerBlock2),
	  threadsPerTask(threadsPerTask), forDapCM(forDapCM)
	{}

	std::string values, noise;
	bool forDapCM;
	std::vector<std::string> wBlockNames;
	std::vector<std::string> hBlockNames;
	boost::numeric::ublas::matrix<unsigned> seeds;
	boost::numeric::ublas::matrix<mf_size_type> nnzPerChunk;
	std::vector<mf_size_type> blockOffsets2;
	int threadsPerTask;
	int chunksPerBlock1;
	int chunksPerBlock2;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & values;
		ar & noise;
		ar & forDapCM;
		ar & seeds;
		ar & nnzPerChunk;
		ar & wBlockNames;
		ar & hBlockNames;
		ar & blockOffsets2;
		ar & threadsPerTask;
		ar & chunksPerBlock1;
		ar & chunksPerBlock2;
	}
};

template<typename M>
void generateRandomFactorsFunction(mf_size_type b1, mf_size_type b2, M& block,
		GenerateRandomFactorsTaskArg arg) {
	// get the relevants
	std::vector<unsigned> localSeeds;
	int startChunk = arg.rowBlocks ? b1*arg.chunksPerBlock : b2*arg.chunksPerBlock;
	for (int b=startChunk ; b<startChunk+arg.chunksPerBlock; b++) {
		localSeeds.push_back(arg.seeds[b]);
	}
	generateFactor(&block, arg.values, localSeeds, arg.rowBlocks, 0, localSeeds.size(), arg.threadsPerTask);
}

template<typename M>
struct GenerateRandomFactorsTask
: public PerBlockTaskVoidArgIndex<M, GenerateRandomFactorsTaskArg, generateRandomFactorsFunction<M>, ID_GENERATE_FACTOR> {
	// typedef PerBlockTaskVoidArgIndex<M, GenerateRandomFactorsTaskArg, generateRandomFactorsFunction<M>, 9999> Task;
};

template<typename M>
void generateRandomDataMatrixFunction(mf_size_type b1, mf_size_type b2, M& block,
		GenerateRandomDataMatrixTaskArg arg) {

	boost::numeric::ublas::matrix<unsigned> localSeeds(arg.chunksPerBlock1,arg.chunksPerBlock2);
	boost::numeric::ublas::matrix<mf_size_type> localNnzPerBlock(arg.chunksPerBlock1,arg.chunksPerBlock2);
	boost::numeric::ublas::matrix<mf_size_type> localNnzStartPoints(arg.chunksPerBlock1,arg.chunksPerBlock2);
	int startChunk1 = b1*arg.chunksPerBlock1 ;
	int startChunk2 = b2*arg.chunksPerBlock2 ;

	mf_size_type startCol = arg.blockOffsets2[b2];
	mf_size_type count = 0;
	for (int i=startChunk1; i<startChunk1+arg.chunksPerBlock1; i++){
		for (int j=startChunk2; j<startChunk2+arg.chunksPerBlock2; j++) {

			localSeeds(i-startChunk1,j-startChunk2) = arg.seeds(i,j)	;
			localNnzPerBlock(i-startChunk1,j-startChunk2) = arg.nnzPerChunk(i,j);
			localNnzStartPoints(i-startChunk1,j-startChunk2) = count;
			count += arg.nnzPerChunk(i,j);

		}
	}

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	int node = tm.world().rank();

	if (arg.forDapCM){
		DenseMatrix* w = mpi2::env().get<DenseMatrix>(arg.wBlockNames[node]);
		DenseMatrixCM* h = mpi2::env().get<DenseMatrixCM>(arg.hBlockNames[b2]);

		generateDataMatrix(&block, w, h, localSeeds, localNnzPerBlock,  0, localSeeds.size1(), 0, localSeeds.size2(),
				localNnzStartPoints, startCol, arg.values, arg.noise, arg.threadsPerTask);
	}else{
		DenseMatrix* w = mpi2::env().get<DenseMatrix>(arg.wBlockNames[b1]);
		DenseMatrixCM* h = mpi2::env().get<DenseMatrixCM>(arg.hBlockNames[node]);

		generateDataMatrix(&block, w, h,localSeeds, localNnzPerBlock, 0, localSeeds.size1(), 0, localSeeds.size2(),
				localNnzStartPoints, startCol, arg.values, arg.noise, arg.threadsPerTask);
	}


}

template<typename M>
struct GenerateRandomDataMatrixTask
: public PerBlockTaskVoidArgIndex<M, GenerateRandomDataMatrixTaskArg,
  generateRandomDataMatrixFunction<M>, ID_GENERATE_DATAMATRIX> {
};


// function

template<typename M>
void generateFactor(DistributedMatrix<M>& m, const std::string& dist, const std::vector<unsigned>& seeds,
		bool rowBlocks, int tasksPerRank) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();

	mf_size_type blocks = rowBlocks ? m.blocks1() : m.blocks2();

	mf_size_type chunks = seeds.size();
	BOOST_ASSERT(chunks % blocks == 0);
	int chunksPerBlock = chunks / blocks;
	BOOST_ASSERT(blocks % tm.world().size() == 0);
	int blocksPerNode = blocks / tm.world().size();

	int threadsPerTask;
	if (blocksPerNode < tasksPerRank) {
		threadsPerTask = tasksPerRank / blocksPerNode;
	} else {
		threadsPerTask = 1;
	}

	GenerateRandomFactorsTaskArg arg(dist, seeds, rowBlocks, chunksPerBlock, threadsPerTask);
	runTaskOnBlocks<GenerateRandomFactorsTask<M> >(m, arg, std::min(tasksPerRank, blocksPerNode));
}

inline std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> generateOriginalFactors(const std::string& wName,
		const std::string& hName, mf_size_type size1, mf_size_type size2, mf_size_type rank, int blocks,
		const std::vector<unsigned>& seedsW, const std::vector<unsigned>& seedsH,
		const std::string& values, int tasksPerRank, bool forDapCM=false){
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	int worldSize= tm.world().size();

	if (!forDapCM){
		DistributedDenseMatrix dW(wName, size1, rank, blocks, 1, true);
		dW.create();
		mf::detail::generateFactor(dW, values, seedsW, true, tasksPerRank);

		std::vector<unsigned> globalSeeds;
		for(int node=0; node<worldSize; node++){
			for (int b2=0; b2<seedsH.size(); b2++){
				globalSeeds.push_back(seedsH[b2]);
			}
		}

		DistributedDenseMatrixCM dH(hName, rank, size2*worldSize, 1, worldSize,  false);
		dH.create();
		mf::detail::generateFactor(dH, values, globalSeeds, false, tasksPerRank);

		std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> p(dW, dH);

		return p;
	} else{

		std::vector<unsigned> globalSeedsW;
		for(int node=0; node<worldSize; node++){
			for (int b1=0; b1<seedsW.size(); b1++){
				globalSeedsW.push_back(seedsW[b1]);
			}
		}

		DistributedDenseMatrix dW(wName, size1*worldSize, rank, worldSize, 1, true);
		dW.create();
		mf::detail::generateFactor(dW, values, globalSeedsW, true, tasksPerRank);

		DistributedDenseMatrixCM dH(hName, rank, size2, 1,blocks,  false);
		dH.create();
		mf::detail::generateFactor(dH, values, seedsH, false, tasksPerRank);

		std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> p(dW, dH);
		return p;
	}

}

template<typename M>
void generateDataMatrix(DistributedMatrix<M>& m, const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		const boost::numeric::ublas::matrix<unsigned>& seeds,
		const boost::numeric::ublas::matrix<mf_size_type>& nnzPerBlock,
		const std::string& values, const std::string& noise,
		int tasksPerRank, bool forDapCM) {

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	int worldSize=tm.world().size();

	mf_size_type blocks1 = m.blocks1();
	mf_size_type blocks2 = m.blocks2() ;
	mf_size_type chunks1 = seeds.size1();
	mf_size_type chunks2 = seeds.size2();
	BOOST_ASSERT(chunks1 % blocks1 == 0);
	BOOST_ASSERT(chunks2 % blocks2 == 0);
	mf_size_type chunksPerBlock1 = chunks1 / blocks1;
	mf_size_type chunksPerBlock2 = chunks2 / blocks2;

	if(forDapCM){
		BOOST_ASSERT(blocks1 % worldSize == 0);
	}else{
		BOOST_ASSERT(w.blocks1() % worldSize == 0);
	}

	int blocks1PerNode = (forDapCM?1:blocks1/worldSize);

	int threadsPerTask;
	if (blocks1PerNode < tasksPerRank) {
		threadsPerTask = tasksPerRank / blocks1PerNode;
	} else {
		threadsPerTask = 1;
	}

	std::vector<std::string> wBlockNames(w.blocks1());
	for(mf_size_type i=0; i<w.blocks1(); i++){
		wBlockNames[i]=w.blocks()(i,0).var();
	}
	std::vector<std::string> hBlockNames(h.blocks2());
	for(mf_size_type i=0; i<hBlockNames.size(); i++){
		hBlockNames[i]=h.blocks()(0,i).var();
	}

	std::vector<mf_size_type> blockOffsets2;
	if (!forDapCM){
		blockOffsets2 = m.blockOffsets2();
	}else{
		blockOffsets2.resize(m.blocks2());
		std::fill(blockOffsets2.begin(), blockOffsets2.end(), 0);
	}

	GenerateRandomDataMatrixTaskArg arg(values, noise, seeds, nnzPerBlock, wBlockNames, hBlockNames, blockOffsets2,
			chunksPerBlock1, chunksPerBlock2, threadsPerTask, forDapCM);
	runTaskOnBlocks<GenerateRandomDataMatrixTask<M> >(m, arg, std::min(tasksPerRank, blocks1PerNode));
}

}//detail

template<typename M>
DistributedMatrix<M> generateFactor(const RandomMatrixDescriptor& f, const std::string& name, mf_size_type blocks1,
		mf_size_type blocks2, bool rowBlocks, int tasksPerRank) {

	if (rowBlocks){
		DistributedMatrix<M> m(name, f.size1, f.rank, blocks1, 1, true);
		m.create();
		mf::detail::generateFactor<M>(m, f.values, f.seedsWorig, true, tasksPerRank);
		return m;
	}else{
		DistributedMatrix<M> m(name, f.rank,f.size2, 1, blocks2, false);
		m.create();
		mf::detail::generateFactor<M>(m, f.values, f.seedsHorig, false, tasksPerRank);
		return m;
	}
}

template<typename M>
void generateDataMatrices(const RandomMatrixDescriptor& f, DistributedMatrix<M>& dv, int tasksPerRank, bool forDapCM,
		DistributedMatrix<M>* dvTest=NULL) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	int worldSize= tm.world().size();

	// first generate the original factors
	std::string wName = "Worig";
	std::string hName = "Horig";

	mf_size_type blocks = (forDapCM?dv.blocks2():dv.blocks1());

	std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> p = mf::detail::generateOriginalFactors(wName,hName,
			f.size1, f.size2,f.rank,blocks,f.seedsWorig,f.seedsHorig,f.values,tasksPerRank,forDapCM);

	DistributedMatrix<DenseMatrix>* dW = &p.first;
	DistributedMatrix<DenseMatrixCM>* dH = &p.second;
	LOG4CXX_INFO(detail::logger,"Original Matrices created...");

	//create v
	dv.create();
	mf::detail::generateDataMatrix(dv, *dW, *dH, f.seedsV, f.nnzPerChunk,
			f.values, f.noise, tasksPerRank, forDapCM);

	//create vTest
	if(dvTest!=NULL){
		dvTest->create();
		std::string noise = "";
		mf::detail::generateDataMatrix(*dvTest, *dW, *dH, f.seedsVtest, f.nnzTestPerChunk,
				f.values, noise, tasksPerRank, forDapCM);
	}

	//release Horig and Worig
	p.first.erase();
	p.second.erase();
}

/*
 * generates or loads the factor matrices from a file(s)
 * TO BE USED FOR EXPERIMENTS
 * */
inline std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> getFactors(const std::string& fileW,
		const std::string& fileH, int tasksPerRank, int worldSize,
		mf_size_type blocks1, mf_size_type blocks2, bool forAsgd){

	if (mf::detail::endsWith(fileW, ".rm")){
		LOG4CXX_INFO(detail::logger, "generating factors on the fly...");
		RandomMatrixDescriptor f;
		f.load(fileW);
		DistributedMatrix<DenseMatrix> dw = generateFactor<DenseMatrix> (f, "W",blocks1,1,
				true, tasksPerRank);
		LOG4CXX_INFO(detail::logger, "Row factor matrix: "
				<< dw.size1() << " x " << dw.size2() << ", " << dw.blocks1() << " x " << dw.blocks2() << " blocks");
		DistributedMatrix<DenseMatrixCM> dh = generateFactor<DenseMatrixCM> (f, "H",1, blocks2,
				false, tasksPerRank);
		LOG4CXX_INFO(detail::logger, "Column factor matrix: "
				<< dh.size1() << " x " << dh.size2() << ", " << dh.blocks1() << " x " << dh.blocks2() << " blocks");
		std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> p(dw, dh);
		return p;

	}else{
		if (forAsgd) tasksPerRank = 1;
		DistributedMatrix<DenseMatrix> dw = loadMatrix<DenseMatrix>(fileW, "W",
				true, tasksPerRank, worldSize, blocks1, 1);
		LOG4CXX_INFO(detail::logger, "Row factor matrix: "
				<< dw.size1() << " x " << dw.size2() << ", " << dw.blocks1() << " x " << dw.blocks2() << " blocks");
		DistributedMatrix<DenseMatrixCM> dh = loadMatrix<DenseMatrixCM>(fileH, "H",
				false, tasksPerRank, worldSize, 1, blocks2);
		LOG4CXX_INFO(detail::logger, "Column factor matrix: "
				<< dh.size1() << " x " << dh.size2() << ", " << dh.blocks1() << " x " << dh.blocks2() << " blocks");
		std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> p(dw, dh);
		return p;
	}
}
/*
 * generates or loads the data / test matrix from a file(s)
 * TO BE USED FOR EXPERIMENTS
 * */
template<typename M>
std::vector<DistributedMatrix<M> > getDataMatrices(const std::string& fileV, const std::string& name, bool partitionByRow,
		int tasksPerRank, int worldSize, mf_size_type blocks1, mf_size_type blocks2, bool forAsgd, bool forDap,
		std::string* fileVtest){
	bool forDapCM = (forDap&&!partitionByRow?true:false);
	std::vector<DistributedMatrix<M> > dataMatrices;
	std::string testName = name+"test";

	if (mf::detail::endsWith(fileV, ".rm")){
		RandomMatrixDescriptor f;
		f.load(fileV);
		LOG4CXX_INFO(detail::logger, "generating data matrices on the fly...");
		DistributedMatrix<M> dv(name, f.size1, f.size2, blocks1, blocks2, partitionByRow);

		if (fileVtest!=NULL){
			if (forDap) blocks2 = blocks1;
			DistributedMatrix<M> dvTest(testName, f.size1, f.size2, blocks1, blocks2, partitionByRow);
			generateDataMatrices(f, dv, tasksPerRank, forDapCM, &dvTest);
			LOG4CXX_INFO(detail::logger, "Test matrix: "
					<< dvTest.size1() << " x " << dvTest.size2() << ", " << nnz(dvTest) << " nonzeros, "
					<< dvTest.blocks1() << " x " << dvTest.blocks2() << " blocks");
			dataMatrices.push_back(dv);
			dataMatrices.push_back(dvTest);
		}else{
			generateDataMatrices(f, dv, tasksPerRank, forDapCM);
			dataMatrices.push_back(dv);
		}

		LOG4CXX_INFO(detail::logger, "Data matrix: "
				<< dv.size1() << " x " << dv.size2() << ", " << nnz(dv) << " nonzeros, "
				<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
	}else{
		if (forAsgd) tasksPerRank = 1;
		DistributedMatrix<M> dv = loadMatrix<M>(fileV,
				name, partitionByRow, tasksPerRank, worldSize, blocks1, blocks2);
		dataMatrices.push_back(dv);

		if (fileVtest!=NULL){
			if (forDap) blocks2 = blocks1;
			DistributedMatrix<M> dvTest = loadMatrix<M>(*fileVtest,
					testName, partitionByRow, tasksPerRank, worldSize,	blocks1, blocks2);
			LOG4CXX_INFO(detail::logger, "Test matrix: "
					<< dvTest.size1() << " x " << dvTest.size2() << ", " << nnz(dvTest) << " nonzeros, "
					<< dvTest.blocks1() << " x " << dvTest.blocks2() << " blocks");
			dataMatrices.push_back(dvTest);
		}

		LOG4CXX_INFO(detail::logger, "Data matrix: "
				<< dv.size1() << " x " << dv.size2() << ", " << nnz(dv) << " nonzeros, "
				<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
	}
	return dataMatrices;
}

}
