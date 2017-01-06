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
#include <queue>

#include <mf/matrix/distributed_matrix.h> // IDE hint

namespace mf {

namespace detail {

template<typename M>
struct CreateMatrixTask {
	static inline std::string id() { return std::string("mf.matrix.CreateMatrixTask/") + mpi2::TypeTraits<M>::name(); }
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::string name;
		mf_size_type size1, size2;
		ch.recv(*mpi2::unmarshal(name, size1, size2));
		M* m = new M(size1, size2);
		mpi2::env().create(name, m);
		boost::mpi::request req = ch.isend();
		mpi2::TaskManager::getInstance().finalizeRequest(req);
	}
};

} // detail

template<typename M>
void DistributedMatrix<M>::create() {
	// this could be parallelized, but it's currently not performance critical...
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	std::vector<boost::mpi::request> reqs;
	for (mf_size_type b1 = 0; b1 < blocks1(); b1++) {
		for (mf_size_type b2 = 0; b2 < blocks2(); b2++) {
			mpi2::RemoteVar rv = block(b1, b2);
			mf_size_type size1 = blockSize1(b1);
			mf_size_type size2 = blockSize2(b2);
			mpi2::Channel ch = tm.spawn<detail::CreateMatrixTask<M> >(rv.rank());
			ch.send( mpi2::marshal(rv.var(), size1, size2) );
			ch.economicRecv(tm.pollDelay());
		}
	}
}

}




