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
#ifndef MF_SGD_FUNCTIONS_UPDATE_LOCK_H
#define MF_SGD_FUNCTIONS_UPDATE_LOCK_H

#include <math.h>

#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_array.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>

namespace mf {

template<typename U>
struct UpdateLock : public UpdateConcept {
	typedef U Update;

	UpdateLock(Update update, mf_size_type size1, mf_size_type size2)
	: update(update), size1(size1), size2(size2), locks1_(new boost::mutex[size1]), locks2_(new boost::mutex[size2]) {
	};

	UpdateLock(Update update, mf_size_type size1, mf_size_type size2, boost::shared_array<boost::mutex>& locks1,
			boost::shared_array<boost::mutex>& locks2)
	: update(update), size1(size1), size2(size2), locks1_(locks1), locks2_(locks2) {
	};

	inline void operator()(FactorizationData<>& data,
			const unsigned i, const unsigned j,	const double x,
			const double eps) {
		BOOST_ASSERT(data.m == size1 && data.n == size2);

		boost::mutex::scoped_lock lock1(locks1_[i]);
		boost::mutex::scoped_lock lock2(locks2_[j]);
		update(data, i, j, x, eps);
	}

	boost::shared_array<boost::mutex>& locks1() { return locks1_; }
	boost::shared_array<boost::mutex>& locks2() { return locks2_; }

private:
	UpdateLock(); // no implementation

	Update update;
	mf_size_type size1;
	mf_size_type size2;
	boost::shared_array<boost::mutex> locks1_;
	boost::shared_array<boost::mutex> locks2_;

	// no serialization!
};

}

#endif
