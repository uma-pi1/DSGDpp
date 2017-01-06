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
 * calculateCSGDblocking.cc
 *
 *  Created on: Jan 31, 2013
 *      Author: chteflio
 */


#include <cmath>
#include <iostream>


using namespace std;


int main(int argc, char *argv[]) {
//	C: cache per core
//	SizeOfInt: size of integer
//	Matrix is  m x n
//	r: rank
//	b: the blocking

	//kdd
	double m = 1000990;
	double n = 624961;
	double N = 252800275;
	int r = 100;	
	long C = 256 * 1024; //Last level cache available in Bytes
	
	int SizeOfInt = 8;


	double alpha = C;
	double beta = (-1)*(m+n)*r*SizeOfInt;
	double gamma = (-1)*N*3*SizeOfInt;


	double delta = beta*beta-4*alpha*gamma;

	double b = ((-1)*beta+sqrt(delta))/(2*alpha);
	
// 	std::cout<<"alpha: "<<alpha<<std::endl;
// 	std::cout<<"beta: "<<beta<<std::endl;
// 	std::cout<<"gamma: "<<gamma<<std::endl;
// 	std::cout<<"delta: "<<delta<<std::endl;

	std::cout<<"Optimal blocking: "<<b<<std::endl;
	

	return 0;
}
