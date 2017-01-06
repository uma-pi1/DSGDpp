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
 * loadDistributedMatrix_impl.h
 *
 *  Created on: Jul 13, 2011
 *      Author: chteflio
 */

#include <mf/matrix/io/loadDistributedMatrix.h>   // compiler hint


namespace mf {

// TODO: we need to make sure that loaded MMC matrices are sorted (call sort() )

namespace detail {

// LOADING

/** Describes an argument for a ReadDistributedMatrixTask in terms of:
 *  (1) the block in which the loaded data will be stored, in the form of a remote variable,
 *  (2) the filename of the file in which the data of this block are stored, and
 *  (3) the format of this file (file-extension)
 */
struct ReadDistributedMatrixTaskArg {
public:
	ReadDistributedMatrixTaskArg() : data(mpi2::UNINITIALIZED) {};

	ReadDistributedMatrixTaskArg(mpi2::RemoteVar block,
			const std::string& filename, const MatrixFileFormat& format)
	: data(block),
	  filename(filename),
	  format(format) {}

	mpi2::RemoteVar data;
	std::string filename;
	MatrixFileFormat format;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & data;
		ar & filename;
		ar & format;
	}
};
/** Constructs an argument for a ReadDistributedMatrixTask.
 *
 * @param b1 the row index of the block
 * @param b2 the column index of the block
 * @param block the block in the form of a remote variable
 * @param f the file descriptor which describes the blocking of the matrix
 *
 */
inline ReadDistributedMatrixTaskArg constructReadDistributedMatrixTaskArg(
		mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
		const BlockedMatrixFileDescriptor& f) {
	return ReadDistributedMatrixTaskArg(block, f.path + f.filenames(b1,b2), f.format);
}
/** Describes a ReadDistributedMatrixTask.
 *  @tparam M type of matrix to be read
 */
template<typename M>
struct ReadDistributedMatrixTask {
	static const std::string id() {	return std::string("__mf/matrix/io/ReadDistributedMatrixTask_") + mpi2::TypeTraits<M>::name();}

	/** runs the ReadDistributedMatrixTask.
	 * It takes a pointer to each block assigned to this task
	 * and loads the data from the respective file to this block.
	 * It assumes that there is already a remote variable for each block at the node.
	 *
	 *  @tparam M type of matrix to be read
	 */
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<ReadDistributedMatrixTaskArg> args;
		ch.recv(args);
		std::vector<boost::mpi::request> reqs(args.size());
		std::vector<int> results(args.size());

		for (unsigned i=0; i<args.size(); i++) {
			/////// if I don't use the DistributedMatrix::create() method I will need to create the variables by myself
			//M* m = new M();
			//mpi2::env().create(args[i].data_.var(), m);
			///////

			// get a pointer pointing at the already reserved space of the node
			M* m=args[i].data.getLocal<M>();
			// read the specific block (stored in filename) into the *m (the already reserved space)
			std::cout << "Reading " << args[i].filename << "..." << std::endl;
			readMatrix(args[i].filename, *m, args[i].format);
			results[i]=1; // later change it to nnz of the block
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};
} // namespace detail

template<typename M>
DistributedMatrix<M> loadMatrix(const BlockedMatrixFileDescriptor& f, const std::string& name,
		bool partitionByRow=true, int tasksPerRank = 1) {
	// take the information needed from fileDescriptor

	mf_size_type blocks1=f.blocks1;
	mf_size_type blocks2=f.blocks2;

	// assign blocks to nodes
	boost::numeric::ublas::matrix<int> blockLocations(blocks1,blocks2);

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();

	computeDefaultBlockLocations(world.size(), blocks1, blocks2, partitionByRow, blockLocations);

	// create the DistributedMatrix...
	DistributedMatrix<M> m(name, f.size1, f.size2,
			f.blockOffsets1, f.blockOffsets2, blockLocations);

	// ... and create remote variables for the blocks at the correct nodes
	m.create();

	// The result is for now 1 (success) or 0 (failure). In the future, it will be the nnz of each block
	boost::numeric::ublas::matrix<int> result;

	// read blocks
	runTaskOnBlocks<M, int, detail::ReadDistributedMatrixTaskArg>(
			m,
			result,
			boost::bind(&detail::constructReadDistributedMatrixTaskArg, _1, _2, _3, boost::cref(f)),
			detail::ReadDistributedMatrixTask<M>::id(),
			tasksPerRank,
			false);

	return m;
}
template<typename M>
DistributedMatrix<M> loadMatrix(const std::string& file,
		const std::string& name, bool partitionByRow, int tasksPerRank, int worldSize,
		mf_size_type blocks1,mf_size_type blocks2,bool test){
	if(mf::detail::endsWith(file, ".rm")){
	//	DistributedMatrix<M> m=generateRandomMatrix<M>(file,name, partitionByRow,tasksPerRank, worldSize,blocks1,blocks2,test);
	}else if (mf::detail::endsWith(file, ".xml")){
		BlockedMatrixFileDescriptor f;
		f.load(file);
		DistributedMatrix<M> m=loadMatrix<M>(f,name,partitionByRow,tasksPerRank);
		return m;
	}else{
		DistributedMatrix<M> m = loadMatrix<M>(name, blocks1, blocks2, partitionByRow, file);
		return m;
	}
}

} // namespace mf

