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
#ifndef MF_MATRIX_IO_WRITE_H
#define MF_MATRIX_IO_WRITE_H

#include <mf/matrix/io/format.h>

namespace mf {

/** Writes a matrix to a file.
 *
 * @param fname file name
 * @param m matrix
 * @param format file format
 * @tparam M matrix type
 */
template<typename M>
void writeMatrix(const std::string& fname, const M& m, MatrixFileFormat format = AUTOMATIC);

} // namespace mf

#include <mf/matrix/io/write_impl.h>

#endif
