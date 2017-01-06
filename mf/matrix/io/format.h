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
#ifndef MF_MATRIX_IO_FORMAT_H
#define MF_MATRIX_IO_FORMAT_H

#define mf_stringify(name) # name

namespace mf {

/** File formats supported by mf library. */
enum MatrixFileFormat {
	AUTOMATIC,                 /**< Automatically choose file format based on extension */
	MM_ARRAY,                  /**< Matrix market array format */
	MM_COORD,                  /**< Matrix market coordinate format */
	BOOST_SPARSE_BIN,          /**< Binary Boost serialization of mf::SparseMatrix (platform-dependent) */
	BOOST_SPARSE_TEXT,         /**< Textual Boost serialization of mf::SparseMatrix (platform-independent) */
	BOOST_DENSE_BIN,           /**< Binary Boost serialization of mf::DenseMatrix (platform-dependent) */
	BOOST_DENSE_TEXT,          /**< Textual Boost serialization of mf::DenseMatrix (platform-independent) */
	MF_INDEX_MAP,			   /**< Textual serialization for vectors of indices of ProjectedSparceMatrices */
	MF_PROJECTED_SPARSE_MATRIX, /**< Textual serialization for Descriptor of ProjectedSparceMatrices */
	MF_RANDOM_MATRIX_FILE
	// ALWAYS ADD NEW FILE FORMATS TO THE END
	// ALSO: UPDATE getName, getExtension, isSparse, getMatrixFormat
};

/** Names of file formats */
inline std::string getName(MatrixFileFormat format) {
	switch (format) {
	case AUTOMATIC: return mf_stringify(AUTOMATIC);
	case MM_ARRAY: return mf_stringify(MM_ARRAY);
	case MM_COORD: return mf_stringify(MM_COORD);
	case BOOST_SPARSE_BIN: return mf_stringify(BOOST_SPARSE_BIN);
	case BOOST_SPARSE_TEXT: return mf_stringify(BOOST_SPARSE_TEXT);
	case BOOST_DENSE_BIN: return mf_stringify(BOOST_DENSE_BIN);
	case BOOST_DENSE_TEXT: return mf_stringify(BOOST_DENSE_TEXT);
	case MF_INDEX_MAP: return mf_stringify(MF_INDEX_MAP);
	case MF_PROJECTED_SPARSE_MATRIX: return mf_stringify(MF_PROJECTED_SPARSE_MATRIX);
	case MF_RANDOM_MATRIX_FILE: return mf_stringify(MF_RANDOM_MATRIX_FILE);
	default:
		RG_THROW(rg::IllegalStateException, "unknown matrix format");
	}
}


namespace detail {
	/** Checks whether string s ends with ending */
	inline bool endsWith(std::string const &s, std::string const &ending)
	{
		if (s.length() > ending.length()) {
			return (0 == s.compare(s.length()-ending.length(), ending.length(), ending));
		} else {
			return false;
		}
	}
}

/** Returns the default file extension for the given format */
inline std::string getExtension(MatrixFileFormat format) {
	switch (format) {
	case MM_ARRAY:
		return "mma";
	case MM_COORD:
		return "mmc";
	case BOOST_SPARSE_BIN:
		return "bsb";
	case BOOST_SPARSE_TEXT:
		return "bst";
	case BOOST_DENSE_BIN:
		return "bdb";
	case BOOST_DENSE_TEXT:
		return "bdt";
	case MF_INDEX_MAP:
		return "mfm";
	case MF_PROJECTED_SPARSE_MATRIX:
		return "mfp";
	case MF_RANDOM_MATRIX_FILE:
			return "rm";
	default:
		RG_THROW(rg::InvalidArgumentException, "no file extension defined for specified matrix format");
	}
}

/** Returns true, if the given format represents a sparse matrix  */
inline bool isSparse(MatrixFileFormat format) {
	switch (format) {
		case MM_ARRAY:
		case BOOST_SPARSE_BIN:
		case BOOST_SPARSE_TEXT:
			return false;
		case MM_COORD:
		case BOOST_DENSE_BIN:
		case BOOST_DENSE_TEXT:
			return true;
		default:
			RG_THROW(rg::InvalidArgumentException, "isSparse() of specified matrix format unknown");
		}
}

/** Returns the format associated with the given extension or filename. */
inline MatrixFileFormat getMatrixFormat(const std::string& name) {
	if (detail::endsWith(name, ".mma") || "mma" == name) return MM_ARRAY;
	if (detail::endsWith(name, ".mmc") || "mmc" == name) return MM_COORD;
	if (detail::endsWith(name, ".bsb") || "bsb" == name) return BOOST_SPARSE_BIN;
	if (detail::endsWith(name, ".bst") || "bst" == name) return BOOST_SPARSE_TEXT;
	if (detail::endsWith(name, ".bdb") || "bdb" == name) return BOOST_DENSE_BIN;
	if (detail::endsWith(name, ".bdt") || "bdt" == name) return BOOST_DENSE_TEXT;
	if (detail::endsWith(name, ".mfm") || "mfm" == name) return MF_INDEX_MAP;
	if (detail::endsWith(name, ".mfp") || "mfp" == name) return MF_PROJECTED_SPARSE_MATRIX;
	if (detail::endsWith(name, ".rm") || "rm" == name) return MF_RANDOM_MATRIX_FILE;
	RG_THROW(rg::InvalidArgumentException, "unknown file ending");
}


}

#endif
