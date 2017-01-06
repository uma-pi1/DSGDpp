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
#include <iostream>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/evaluation.h>
#include <util/io.h>

#include <mf/mf.h>

using namespace std;
using namespace mf;
using namespace rg;
using namespace boost::numeric::ublas;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));


#define NDIM 4

int main() {
	mf_size_type n = 4;

	// coefficient matrix
	DenseMatrixCM a(n, n), aLu(n,n), aQr(n,n), aS(n,n);
	a(0,0) = 1.0;
	a(0,1) = -1.0;
	a(0,2) = 2.0;
	a(0,3) = -1.0;
	a(1,0) = 2.0;
	a(1,1) = -2.0;
	a(1,2) = 3.0;
	a(1,3) = -3.0;
	a(2,0) = 1.0;
	a(2,1) = 1.0;
	a(2,2) = 1.0;
	a(2,3) = 0.0;
	a(3,0) = 1.0;
	a(3,1) = -1.0;
	a(3,2) = 4.0;
	a(3,3) = 3.0;

	// right hand side
	boost::numeric::ublas::vector<double> b(n), x(n), tau(n);
	b[0] = -8.0;
	b[1] = -20.0;
	b[2] = -2.0;
	b[3] = 4.0;

	// print
	cout << "A = " << a << endl;
	cout << "b = " << b << endl;

	// compute LU factorization
	boost::numeric::ublas::vector<clapack::integer> ipiv(n);
	aLu = a;
	cout << lu(aLu, ipiv) << endl;
	cout << "A(LU) = " << aLu << endl;
	cout << "Row permutations = " << ipiv << endl;

	// solve LP
	x = b;
	cout << lpLu(aLu, ipiv, &x.data()[0]) << endl;

	// print
	cout << "x = " << x << endl;
	cout << "Ax = " << prod(a, x) << endl;

	// qr decomposition
	aQr = a;
	cout << qr(aQr, tau) << endl;
	cout << "A(QR) = " << aQr << endl;
	cout << "tau = " << tau << endl;

	// solve least squares problem
	std::pair<clapack::integer, clapack::integer> swork = llsWork(n,n);
	boost::numeric::ublas::vector<double> s(n); // singular values
	boost::numeric::ublas::vector<double> work(swork.first);
	boost::numeric::ublas::vector<clapack::integer> iwork(swork.second);
	aS = a;
	x = b;
	cout << lls(aS, x.data().begin(), s, work, iwork) << endl;
	cout << "A(S) = " << aS << endl;
	cout << "s = " << s << endl;
	cout << "x = " << x << endl;
	cout << "Ax = " << prod(a, x) << endl;

	return 0;
}
