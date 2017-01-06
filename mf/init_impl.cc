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
#include <mf/init.h>

#include <mpi2/mpi2.h>

#include <mf/types.h>
#include <mf/matrix/op/project.h>
#include <mf/register/register.h>
#include <mf/register/register-generated.h>

namespace mf {
boost::mpi::communicator& mfInit(int& argc, char**& argv) {
	boost::mpi::communicator& world = mpi2::mpi2init(argc, argv);
	detail::registerMatrixTasks();
	detail::registerGeneratedMatrixTasks();
	mpi2::registerTypes<MfBuiltinTypes>();
	mpi2::registerType<ProjectedSparseMatrix>();
	return world;
}

void mfStart(){
	mpi2::mpi2start();
}

void mfStop() {
	mpi2::mpi2stop();
}
void mfFinalize() {
	mpi2::mpi2finalize();
}


} // namespace mf

