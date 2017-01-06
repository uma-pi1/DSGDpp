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
 * ioProjected.h
 *
 *  Created on: Jul 14, 2011
 *      Author: chteflio
 */

#ifndef MF_MF_MATRIX_IO_IOPROJECTED_H_
#define MF_MF_MATRIX_IO_IOPROJECTED_H_

#include <vector>

#include <mf/matrix/op/project.h>
#include <mf/matrix/io/mappingDescriptor.h>

namespace mf {

/*
 *  Writes the projected Matrix to disk
 *  Information about the names of the files are taken from the mapDescriptor
 *
 *	@param 	mapDescriptor the file descriptor
 *	@param	projectedMatrix the projected matrix to be written
 */
void writeProjectedMatrix(const  ProjectedSparseMatrix& projectedMatrix, const IndexMapFileDescriptor& mapDescriptor);

/*
 *  reads the projected Matrix from disk
 *
 *	@param 	projectedMatrixFilename the name of the file containing the data
 *	@param	map1Filename and map2Filename the names of the files containing the mappings, map1 and map2
 * 	@param[out]	projectedMatrix the projected matrix
 */
void readProjectedMatrix(ProjectedSparseMatrix& projectedMatrix, const std::string& projectedMatrixFilename,
		 const std::string& map1Filename, const std::string& map2MappingFilename);


/*
 *  reads the projected Matrix from disk
 *  Information about the names of the files are taken from the IndexMapDescriptor,
 *  which is stored in the file indexMapDescriptorFilename
 *
 *	@param 	indexMapDescriptorFilename the name of the file containing the file descriptor
 *	@param[out]	projectedMatrix the projected matrix
 */
void readProjectedMatrix(ProjectedSparseMatrix& projectedMatrix,const std::string& indexMapDescriptorFilename);

/*
 *  reads the mappings of rows or columns between the projected and the original matrix
 *
 * 	@param 	file the name of the mfm file containing the mappings
 * 	@param[out] map the resulting mapping vector
 *	@param[out] size the size of the original matrix
 */
void readMapFile(const std::string& file, std::vector<mf_size_type>& map, mf_size_type& size);

/*
 *  writes the mappings of rows or columns between the projected and the original matrix
 *
 * 	@param 	file the name of the mfm file containing the mappings
 *	@param	map the mapping vector
 *	@param 	size the size of the original matrix
 */
void writeMapFile(const std::string& file, const std::vector<mf_size_type>& map, mf_size_type size);
}


#endif /* MF_MF_MATRIX_IO_IOPROJECTED_H_ */
