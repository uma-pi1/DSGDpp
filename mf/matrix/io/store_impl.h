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
#include <mf/matrix/io/store.h>   // compiler hint

namespace mf {

namespace detail {

// Arguments for each block is block number, location of data, and filename
struct WriteDistributedMatrixTaskArg {
public:
	WriteDistributedMatrixTaskArg() : data_(mpi2::UNINITIALIZED) {};

	template<typename M>
	WriteDistributedMatrixTaskArg(const DistributedMatrix<M>& m, mf_size_type b1, mf_size_type b2,
			const std::string& filename, MatrixFileFormat format)
	: b1_(b1), b2_(b2), data_(m.block(b1,b2)), filename_(filename), format_(format) {
	}

	mf_size_type b1_;
	mf_size_type b2_;
	mpi2::RemoteVar data_;
	std::string filename_;
	MatrixFileFormat format_;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & b1_;
		ar & b2_;
		ar & data_;
		ar & filename_;
		ar & format_;
	}
};

template<typename M>
WriteDistributedMatrixTaskArg constructWriteDistributedMatrixTaskArg(
		mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
		const DistributedMatrix<M>& m, const BlockedMatrixFileDescriptor& f) {
	return WriteDistributedMatrixTaskArg(m, b1, b2, f.path + f.filenames(b1,b2), f.format);
}

template<typename M>
struct WriteDistributedMatrixTask {
	static const std::string id() { return std::string("__mf/matrix/io/WriteDistributedMatrixTask_") + mpi2::TypeTraits<M>::name(); }
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<WriteDistributedMatrixTaskArg> args;
		ch.recvAsync(args);
		std::vector<boost::mpi::request> reqs(args.size());
		std::vector<std::string> results(args.size());
		for (unsigned i=0; i<args.size(); i++) {
			const M* m = args[i].data_.getLocal<M>();
			std::cout << "Writing " << args[i].filename_ << "..." << std::endl;
			writeMatrix(args[i].filename_, *m, args[i].format_);
			results[i] = args[i].filename_;
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

} // namespace detail

template<typename M>
void storeMatrix(const DistributedMatrix<M>& m, const BlockedMatrixFileDescriptor& f,
		int tasksPerRank) {

	// write blocks
	boost::numeric::ublas::matrix<std::string> result;
	runTaskOnBlocks<M, std::string, detail::WriteDistributedMatrixTaskArg>(
			m,
			result,
			boost::bind(&detail::constructWriteDistributedMatrixTaskArg<M>, _1, _2, _3, boost::cref(m), boost::cref(f)),
			detail::WriteDistributedMatrixTask<M>::id(),
			tasksPerRank,
			false); // no asynchronous receive for strings
}

} // namespace mf
