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
#include <boost/archive/text_oarchive.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/serialization/serialization.hpp>

#include <mf/matrix/io/descriptor.h>

namespace mf {

/** Stores the descriptor as XML file. */
void BlockedMatrixFileDescriptor::load(const std::string &filename) {
	using boost::property_tree::ptree;
	ptree pt;
	read_xml(filename, pt);

	size1 = pt.get<mf_size_type>("matrix.size1");
	size2 = pt.get<mf_size_type>("matrix.size2");
	blocks1 = pt.get<mf_size_type>("matrix.blocks1");
	blocks2 = pt.get<mf_size_type>("matrix.blocks2");
	path = pt.get<std::string>("matrix.path");
	blockOffsets1.clear();
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("matrix.blockOffsets1"))
		blockOffsets1.push_back(v.second.get_value<mf_size_type>());
	blockOffsets2.clear();
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("matrix.blockOffsets2"))
		blockOffsets2.push_back(v.second.get_value<mf_size_type>());
	filenames.resize(blocks1, blocks2, false);
	mf_size_type i=0, j=0;
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("matrix.filenames")) {
		filenames(i, j) = v.second.get_value<std::string>();
		j++;
		if (j == blocks2) {
			i++;
			j = 0;
		}
	}
	format = getMatrixFormat(pt.get<std::string>("matrix.format"));
}

/** Reads a descriptor from an XML file. */
void BlockedMatrixFileDescriptor::save(const std::string &filename) {
	using boost::property_tree::ptree;
	ptree pt;
	pt.put("matrix.size1", size1);
	pt.put("matrix.size2", size2);
	pt.put("matrix.blocks1", blocks1);
	pt.put("matrix.blocks2", blocks2);
	BOOST_FOREACH(const mf_size_type &blockOffset1, blockOffsets1) {
		pt.add("matrix.blockOffsets1.offset", blockOffset1);
	}
	BOOST_FOREACH(const mf_size_type &blockOffset2, blockOffsets2)
		pt.add("matrix.blockOffsets2.offset", blockOffset2);
	pt.put("matrix.path", path);
	for (mf_size_type i = 0; i<blocks1; i++) {
		for (mf_size_type j = 0; j<blocks2; j++) {
			pt.add("matrix.filenames.filename", filenames(i,j));
		}
	}
	pt.put("matrix.format", getExtension(format));
	write_xml(filename, pt, std::locale(), boost::property_tree::xml_parser::xml_writer_make_settings<std::basic_string<char> >(' ', 4));
}


}

