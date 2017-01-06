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
 * mappingDescriptor.h
 *
 *  Created on: Jul 14, 2011
 *      Author: chteflio
 */

#ifndef MF_MF_MATRIX_IO_MAPPINGDESCRIPTOR_H_
#define MF_MF_MATRIX_IO_MAPPINGDESCRIPTOR_H_

#include <boost/filesystem.hpp>
#include <util/exception.h>
#include <util/io.h>
#include <mf/matrix/io/format.h>

namespace mf {

/** Describes a ProjectedSparceMatrix by describing:
 * i) 	the file where the original matrix is stored (field: matrixFilename)
 * ii)	the file where the projected matrix is stored (field: projectedMatrixFilename)
 * iii) the file where the mappings of rows, between projected and original matrix, are stored
 * 		(field: map1Filename)
 * iv) the file where the mappings of columns, between projected and original matrix, are stored
 * 		(field: map2Filename)
 */

struct IndexMapFileDescriptor{
	IndexMapFileDescriptor(const std::string& mfpFilename, const std::string& matrixFilename = "")
	: matrixFilename(matrixFilename) {
		if (getMatrixFormat(mfpFilename) != MF_PROJECTED_SPARSE_MATRIX) {
			RG_THROW(rg::InvalidArgumentException, "Invalid filename");
		}

		boost::filesystem::path file(mfpFilename);
		std::string filename = file.stem().string();
		std::string extension = file.extension().string();

		path = file.parent_path().string();
		projectedMatrixFilename = rg::paste(filename, ".", getExtension(MM_COORD));
		map1Filename = rg::paste(filename, "-map1.", getExtension(MF_INDEX_MAP));
		map2Filename = rg::paste(filename, "-map2.", getExtension(MF_INDEX_MAP));
	}

	IndexMapFileDescriptor(const std::string& path,
			const std::string& matrixFilename,
			const std::string& projectedMatrixFilename,
			const std::string& map1Filename,
			const std::string& map2Filename)
	:path(path),matrixFilename(matrixFilename),projectedMatrixFilename(projectedMatrixFilename),
	 map1Filename(map1Filename),map2Filename(map2Filename){}

	IndexMapFileDescriptor() {
		// for loading the descriptor later from a file
	}

	/** Reads the descriptor from an XML file. */
    void load(const std::string &filename);

    /** Saves the descriptor in an XML file. */
    void save(const std::string &filename);

    static std::string compose(std::string f1, std::string f2) {
    	boost::filesystem::path p1(f1);
    	boost::filesystem::path p2(f2);
    	if (p2.is_absolute()) {
    		return p2.string();
    	} else {
    		return (p1 /= p2).string();
    	}
    }

    std::string completeProjectedMatrixFilename() const {
    	return compose(path, projectedMatrixFilename);
    }

    std::string completeMap1Filename() const {
    	return compose(path, map1Filename);
    }

    std::string completeMap2Filename() const {
    	return compose(path, map2Filename);
    }

    std::string path;
	std::string matrixFilename;
	std::string projectedMatrixFilename;
	std::string map1Filename;
	std::string map2Filename;
};

} // namespace mf



#endif /* MF_MF_MATRIX_IO_MAPPINGDESCRIPTOR_H_ */
