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
 * seedDescriptor_impl.cc
 *
 *  Created on: May 10, 2012
 *      Author: chteflio
 */
#include <boost/archive/text_oarchive.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/serialization/serialization.hpp>

#include <mf/matrix/io/randomMatrixDescriptor.h>

namespace mf {

/** Stores the descriptor as XML file. */
void RandomMatrixDescriptor::load(const std::string &filename) {
	using boost::property_tree::ptree;
	ptree pt;
	read_xml(filename, pt);

	size1 = pt.get<mf_size_type>("size1");
	size2 = pt.get<mf_size_type>("size2");
	chunks1 = pt.get<mf_size_type>("blocks1");
	chunks2 = pt.get<mf_size_type>("blocks2");
	nnz=pt.get<mf_size_type>("nnz");
	nnzTest=pt.get<mf_size_type>("nnzTest");
	values=pt.get<std::string>("values");
	noise=pt.get<std::string>("noise");
	rank=pt.get<mf_size_type>("rank");

	seedsWorig.clear();
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("Worig"))
		seedsWorig.push_back(v.second.get_value<unsigned>());

	seedsHorig.clear();
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("Horig"))
		seedsHorig.push_back(v.second.get_value<unsigned>());

	if (nnz!=0 && nnzTest!=0){
		seedsV.resize(chunks1, chunks2, false);
		mf_size_type i=0, j=0;
		BOOST_FOREACH(ptree::value_type &v, pt.get_child("V.seeds")) {
			seedsV(i, j) = v.second.get_value<unsigned>();
			j++;
			if (j == chunks2) {
				i++;
				j = 0;
			}
		}

		nnzPerChunk.resize(chunks1, chunks2, false);
		i=0, j=0;
		BOOST_FOREACH(ptree::value_type &v, pt.get_child("V.nnzs")) {
			nnzPerChunk(i, j) = v.second.get_value<mf_size_type>();
			j++;
			if (j == chunks2) {
				i++;
				j = 0;
			}
		}

		seedsVtest.resize(chunks1, chunks2, false);
		i=0, j=0;
		BOOST_FOREACH(ptree::value_type &v, pt.get_child("Vtest.seeds")) {
			seedsVtest(i, j) = v.second.get_value<unsigned>();
			j++;
			if (j == chunks2) {
				i++;
				j = 0;
			}
		}
		nnzTestPerChunk.resize(chunks1, chunks2, false);
		i=0, j=0;
		BOOST_FOREACH(ptree::value_type &v, pt.get_child("Vtest.nnzs")) {
			nnzTestPerChunk(i, j) = v.second.get_value<mf_size_type>();
			j++;
			if (j == chunks2) {
				i++;
				j = 0;
			}
		}
	}

}

/** Reads a descriptor from an XML file. */
void RandomMatrixDescriptor::save(const std::string &filename) {
	using boost::property_tree::ptree;
	ptree pt;
	pt.put("size1", size1);
	pt.put("size2", size2);
	pt.put("blocks1", chunks1);
	pt.put("blocks2", chunks2);
	pt.put("nnz", nnz);
	pt.put("nnzTest", nnzTest);
	pt.put("values", values);
	pt.put("noise", noise);
	pt.put("rank", rank);


	for (mf_size_type i = 0; i<chunks1; i++) {
		pt.add("Worig.seed", seedsWorig[i]);
	}
	for (mf_size_type j = 0; j<chunks2; j++) {
		pt.add("Horig.seed", seedsHorig[j]);
	}

	if (nnz!=0 && nnzTest!=0){
		for (mf_size_type i = 0; i<chunks1; i++) {
			for (mf_size_type j = 0; j<chunks2; j++) {
				pt.add("V.seeds.seed", seedsV(i,j));
				pt.add("V.nnzs.nnz",nnzPerChunk(i,j));
			}
		}
		for (mf_size_type i = 0; i<chunks1; i++) {
			for (mf_size_type j = 0; j<chunks2; j++) {
				pt.add("Vtest.seeds.seed", seedsVtest(i,j));
				pt.add("Vtest.nnzs.nnz",nnzTestPerChunk(i,j));
			}
		}
	}


	write_xml(filename, pt, std::locale(), boost::property_tree::xml_parser::xml_writer_make_settings(' ', 4));
}

void RandomMatrixDescriptor::calculateNnzPerBlock(boost::numeric::ublas::matrix<mf_size_type>& nnzPerChunk, mf_size_type nnz,
		mf_size_type chunks1, mf_size_type chunks2){
	mf_size_type nnzRemaining = nnz;
	mf_size_type blockCounter = 0;
	if (nnz >= INT_MAX) {
		for (mf_size_type i = 0; i < nnzPerChunk.size1(); i++) {
			for (mf_size_type j = 0; j < nnzPerChunk.size2(); j++) {
				rg::Random32 random (this->seedsV(i,j));
				double prob = 1.0 / ((chunks1 * chunks2) - blockCounter);
				double mean = nnzRemaining * prob;
				double sigma = sqrt(nnzRemaining * prob * (1. - prob));
				boost::normal_distribution<> dist(mean, sigma);
				boost::variate_generator<rg::Random32::Prng&, boost::normal_distribution<> > gen(random.prng(), dist);
				mf_size_type nnzBlock = gen();
				nnzPerChunk(i,j) = nnzBlock;
				nnzRemaining -= nnzBlock;
				blockCounter++;
			}
		}
	} else {
		for (mf_size_type i = 0; i < nnzPerChunk.size1(); i++) {
			for (mf_size_type j = 0; j < nnzPerChunk.size2(); j++) {
				rg::Random32 random (this->seedsV(i,j));
				double prob = 1.0 / ((chunks1 * chunks2) - blockCounter);
				boost::binomial_distribution<> dist (nnzRemaining, prob);
				boost::variate_generator<rg::Random32::Prng&, boost::binomial_distribution<> > gen(random.prng(), dist);
				mf_size_type nnzBlock = gen();
				nnzPerChunk(i,j) = nnzBlock;
				nnzRemaining -= nnzBlock;
				blockCounter++;
			}
		}
	}
}
}





