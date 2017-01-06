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
 * Examples of using user-defined types with mpi2. Also illustrates use of special
 * "non-initializing" constructor serialization.
 */
#include <iostream>
#include <string>

#include <util/io.h>

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

// a user-defined type
class MyType {
public:
	// a special constructor that does not initialize the class
	MyType(SerializationConstructor _) { }

	// the standard constructor
	MyType(int a, int b) : a(a), b(b) {
	}

private:
	int a;
	int b;

	// use boost serialization to serialize this class
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & a;
		ar & b;
	}

	// for pretty printing
	template<typename CharT, typename Traits>
	friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, const MyType& t);
};

// pretty printing
template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, const MyType& t)
{
	return out << "(a: " << t.a << ", b: " << t.b << ")";
}

// this registers type traits for the type (including the type's name)
MPI2_TYPE_TRAITS(MyType);

// this registers the serialization constructor with mpi2
// (only needed when there is no default constructor or different semantics needed)
MPI2_SERIALIZATION_CONSTRUCTOR(MyType);

int main(int argc, char *argv[]) {
	boost::mpi::communicator& world = mpi2init(argc, argv);
	if (world.size() < 2) {
		cerr << "ERROR: You need to run 'udt' on at least 2 ranks!" << endl;
		cerr << "Try 'mpirun -np 2 " << argv[0] << "'" << endl;
		return(-1);
	}

	// register MyType with mpi2
	registerType<MyType>();

	// fire up mpi2
	mpi2start();
	boost::this_thread::sleep(boost::posix_time::milliseconds(100)); // just to ensure output in right order

	if (world.rank() == 0) {
		// create an instance
		MyType t(1,2);

		// store it at rank 1
		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Storing value " << t << " at rank 1");
		RemoteVar v = createCopy(1, "value", t);

		// receive it from rank 1
		LOG4CXX_INFO(logger, "Fetching value from rank 1");
		MyType tCopy(UNINITIALIZED); // use the non-initializing constructor
		// Alternative: MyType* tCopy = Mpi2Constructor<MyType>::construct()
		//              this would works for classes w/o initialization constructor!
		v.getCopy(tCopy);
		LOG4CXX_INFO(logger, "Received " << tCopy);
		LOG4CXX_INFO(logger, "");
	}

	// shut down
	logger->info("");
	mpi2stop();
	mpi2finalize();

	return 0;
}
