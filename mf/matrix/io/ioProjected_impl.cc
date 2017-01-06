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
 * ioProjected_impl.h
 *
 *  Created on: Jul 14, 2011
 *      Author: chteflio
 */

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <mf/matrix/io/ioProjected.h>

#include <mf/matrix/io/read.h>
#include <mf/matrix/io/write.h>


namespace mf
{

void writeMapFile(const std::string& file,const std::vector<mf_size_type>& map, mf_size_type size){
	LOG4CXX_INFO(detail::logger, "Writing map file:  " << file);

	std::ofstream out(file.c_str());
	if (!out.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + file);

	// write dimension line
	out << map.size() << " " << size <<std::endl;
	mf_size_type i=0;

	// write elements
	BOOST_FOREACH( mf_size_type element, map) {
		out <<(i+1)<<" "<<(element+1)<<std::endl;
		i++;
	}

	// done
	out.close();
}

void readMapFile(const std::string& file, std::vector<mf_size_type>& map, mf_size_type& size){

	LOG4CXX_INFO(detail::logger, "Reading map file:  " << file);

	// open file
	std::ifstream in(file.c_str());
	if (!in.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + file);

	std::string line;
	mf_size_type projectedSize;

	if (!getline(in, line))
		RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + file);

	// read 1st line, which contains the dimensions of sample and original matrix
	if (sscanf(line.c_str(), "%ld %ld", &projectedSize, &size) != 2)
			RG_THROW(rg::IOException, std::string("Invalid vector dimensions in file ") + file);

	// initialize the vector of indexes
	map.reserve(projectedSize);


	// read matrix
	for (mf_size_type p=0; p<projectedSize; p++){
		if (!getline(in, line)) RG_THROW(rg::IOException, std::string("Unexpected EOF in file ") + file);
		mf_size_type indSample, indOriginal;

		if (sscanf(line.c_str(), "%ld %ld", &indSample, &indOriginal) != 2)
			RG_THROW(rg::IOException, std::string("Invalid index entry in file ") + file);
		map.push_back(indOriginal-1); // row1 should map to array[0]

	}

	// done
	in.close();
}


void writeProjectedMatrix(const ProjectedSparseMatrix& projectedMatrix, const IndexMapFileDescriptor& mapDescriptor){
	LOG4CXX_INFO(detail::logger, "Writing projected matrix & map files: " <<
			mapDescriptor.completeProjectedMatrixFilename() << " & " <<
			mapDescriptor.completeMap1Filename() << ", " <<
			mapDescriptor.completeMap2Filename()
	);
	writeMatrix(mapDescriptor.completeProjectedMatrixFilename(), projectedMatrix.data);
	writeMapFile(mapDescriptor.completeMap1Filename(), projectedMatrix.map1, projectedMatrix.size1);
	writeMapFile(mapDescriptor.completeMap2Filename(), projectedMatrix.map2, projectedMatrix.size2);

}

void readProjectedMatrix(ProjectedSparseMatrix& projectedMatrix, const std::string& projectedMatrixFilename,
		 const std::string& map1Filename, const std::string& map2Filename){
	readMatrix(projectedMatrixFilename, projectedMatrix.data);
	readMapFile(map1Filename, projectedMatrix.map1, projectedMatrix.size1);
	readMapFile(map2Filename, projectedMatrix.map2, projectedMatrix.size2);
}

void readProjectedMatrix(ProjectedSparseMatrix& projectedMatrix, const  std::string& indexMapDescriptorFilename){
	IndexMapFileDescriptor mapDescriptor;
	mapDescriptor.load(indexMapDescriptorFilename);
	readProjectedMatrix(projectedMatrix, mapDescriptor.completeProjectedMatrixFilename(),
			mapDescriptor.completeMap1Filename(),  mapDescriptor.completeMap2Filename());
}

}
