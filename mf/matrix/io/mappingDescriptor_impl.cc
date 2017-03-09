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
 * mappingDescriptor_impl.h
 *
 *  Created on: Jul 14, 2011
 *      Author: chteflio
 */

#include <boost/archive/text_oarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/serialization/serialization.hpp>

#include <mf/logger.h>
#include <mf/matrix/io/mappingDescriptor.h>

namespace mf
{

void IndexMapFileDescriptor::load(const std::string &filename){
    using boost::property_tree::ptree;
    ptree pt;
    read_xml(filename, pt);

	LOG4CXX_INFO(detail::logger, "Reading MFP descriptor " << filename);
    matrixFilename = pt.get<std::string>("files.matrixFilename");
    projectedMatrixFilename = pt.get<std::string>("files.projectedMatrixFilename");
    map1Filename = pt.get<std::string>("files.map1Filename");
    map2Filename = pt.get<std::string>("files.map2Filename");

    boost::filesystem::path f(filename);
    path = f.parent_path().string();
}


void IndexMapFileDescriptor::save(const std::string &filename) {
	LOG4CXX_INFO(detail::logger, "Writing MFP descriptor " << filename);

    using boost::property_tree::ptree;
    ptree pt;

    pt.put("files.matrixFilename", matrixFilename);
    pt.put("files.projectedMatrixFilename", projectedMatrixFilename);
    pt.put("files.map1Filename", map1Filename);
    pt.put("files.map2Filename",map2Filename);

    write_xml(filename, pt, std::locale(), boost::property_tree::xml_parser::xml_writer_make_settings<std::basic_string<char> >(' ', 4));
}

}
