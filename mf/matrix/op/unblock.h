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
#ifndef MF_MATRIX_OP_UNBLOCK_H
#define MF_MATRIX_OP_UNBLOCK_H

namespace mf {

/** Fetches and combines the blocks of a distributed matrix into a non-distributed matrix.
 * Note that this method is memory intensive (the result will be stored on just one node) and
 * that its current implementation is rather inefficient (as its rarely needed).
 *
 * @param source distributed input matrix
 * @param[out] target output matrix
 * @tparam Min type of input matrix blocks
 * @tparam Mout type of output matrix
 */
template<typename Min, typename Mout>
void unblock(const DistributedMatrix<Min>& source, Mout& target) {
	// TODO: efficiency
	target.resize(source.size1(), source.size2(), false);
	Min temp(source.blockSize1(0), source.blockSize2(0)); // sizes will be overwritten automatically
	for (mf_size_type b1=0; b1<source.blocks1(); b1++) {
		for (mf_size_type b2=0; b2<source.blocks2(); b2++) {
			Min* block;
			mpi2::RemoteVar var = source.block(b1,b2);
			if (var.isLocal()) {
				block = var.getLocal<Min>();
			} else {
				var.getCopy(temp);
				block = &temp;
			}
			boost::numeric::ublas::subrange(target,
					source.blockOffset1(b1), source.blockOffset1(b1)+source.blockSize1(b1),
					source.blockOffset2(b2), source.blockOffset2(b2)+source.blockSize2(b2))
				= *block;
		}
	}
}

namespace detail {
	template<typename M>
	struct UnblockTask {
		static const std::string id() { return std::string("__mf/matrix/UnblockTask_") + mpi2::TypeTraits<M>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			// which matrix to fetch
			DistributedMatrix<M> dm(mpi2::UNINITIALIZED);
			std::string name;
			ch.recv(*mpi2::unmarshal(dm, name));
			M& target = *mpi2::env().get<M>(name);
			unblock(dm, target);
			ch.send();
		}
	};
}

/** Unblocks the given matrix and stores the result in the environment of every node.
 * The variable in the environment must exist already and be of the correct type.
 *
 * @param in matrix to unblock
 * @param out name of variable that should store the result (on all nodes)
 */
template<typename M>
void unblockAll(DistributedMatrix<M> in, const std::string& out) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	std::vector<mpi2::Channel> channels;
	tm.spawnAll<detail::UnblockTask<M> >(channels);
	mpi2::sendAll(channels, mpi2::marshal(in, out));
	mpi2::recvAll(channels);
}

}

#endif
