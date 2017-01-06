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
/** \file
 *
 * Parsing command line arguments for mf tools.
 */
#ifndef MF_TOOLS_PARSE_H
#define MF_TOOLS_PARSE_H

// added by me
#include <mf/mf.h>

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace parse {

using namespace boost;
using namespace std;
// added by me
using namespace mf;

template<typename T>
void parseArg(const string& argName, const string& arg,
		string& name, std::vector<T>& args) {
	smatch what;
	if (!regex_match(arg, what, regex("^(.*)\\((.*)\\)$"), boost::match_default)) {
		args.clear();
		name = arg;
	} else {
		unsigned argNo = 0;
		try {
			name = what.str(1);
			string argsString = what.str(2);
			std::vector<string> argsSplit;
			split(argsSplit, argsString, is_any_of(","));
			args.clear();
			for (; argNo<argsSplit.size(); argNo++) {
				args.push_back(lexical_cast<T>(argsSplit[argNo]));
			}
		} catch(bad_lexical_cast &) {
			std::cerr << "Invalid argument " << (argNo+1) << " in " << arg << std::endl;
			exit(1);
		}
	}
}

template <typename Out>
void printArg(Out& out, const string& name, const std::vector<string>& args) {
	out << name << "(";
	string sep="";
	for (unsigned i=0; i<args.size(); i++) {
		out << sep << args[i];
		sep = ",";
	}
	out << ")";
}

template<typename F>
void parseDistribution(const string& argName, const string& arg, F f) {
	string name;
	std::vector<string> args;
	parseArg(argName, arg, name, args);
	try {
		if (name.compare("Normal") == 0) {
			if (args.size() != 2) {
				goto error;
			}
			double mean = lexical_cast<double>(args[0]);
			double sigma = lexical_cast<double>(args[1]);
			//cout << "Distribution: " << "Normal(" << mean << "," << sigma << ")" << endl;
			f(boost::normal_distribution<>(0, sigma));
			return;
		} else if (name.compare("Uniform") == 0) {
			if (args.size() != 2) {
				goto error;
			}
			double min = lexical_cast<double>(args[0]);
			double max = lexical_cast<double>(args[1]);
			//cout << "Distribution: " << "Uniform(" << min << "," << max << ")" << endl;
			f(boost::uniform_real<>(min, max));
			return;
		}
	} catch(bad_lexical_cast &) {
	}

	error:
	cerr << "Error in argument '" << argName << "': Unknown or invalid distribution '";
	printArg(cerr, name, args);
	cerr << "'" << endl;
	cerr << "Valid choices are: " << endl;
	cerr << "    Normal(<mean>,<stddev>)" << endl;
	cerr << "    Uniform(<min>,<max>)" << endl;
	exit(1);
}

/*********************************************************************************/

template<typename F>
void parseTruncate(const string& argName, const string& arg, F& f) {
	string name;
	std::vector<string> arguments;
	parseArg(argName, arg, name, arguments);
	try {
		if (name.compare("") == 0) {
			if (arguments.size() != 2) {
				goto error;
			}
			f.truncateArgs.resize(2);
			f.truncateArgs[0] = lexical_cast<double>(arguments[0]);
			f.truncateArgs[1] = lexical_cast<double>(arguments[1]);
			return;
		}
	}
	catch(bad_lexical_cast &) {
	}
	error:
	cerr << "Error in argument " << argName << " with options ";
	parse::printArg(cerr, name, arguments);
	cerr << endl;
	cerr << "Valid options are:" << endl;
	cerr << "    (<min>,<max>)" << endl;
	exit(1);
}

template<typename F>
void parseDecay(const string& argName, const string& arg, F& f) {
	string name;
	std::vector<string> arguments;
	parseArg(argName, arg, name, arguments);
	try {
		if (name.compare("Const") == 0) {
			if (arguments.size() == 1) {
				double epsilon = lexical_cast<double>(arguments[0]);
				cout << "Decay: " << "Const(" << epsilon << ")" << endl;
				f.decayName = "Const";
				f.decayArgs = arguments;
				return;
			} else {
				goto error;
			}
		} else if (name.compare("Auto") == 0) {
			if (arguments.size() == 3) {
				double epsilon = lexical_cast<double>(arguments[0]);
				unsigned tries = lexical_cast<unsigned>(arguments[1]);
				string inputSampleMatrixFile = lexical_cast<string>(arguments[2]);
				f.epsilon = epsilon;
				f.tries = tries;
				f.inputSampleMatrixFile = inputSampleMatrixFile;
				f.decayName = "Auto";
				f.decayArgs = arguments;
				return;
			} else {
				goto error;
			}
		} else if (name.compare("BoldDriver") == 0) {
			if (arguments.size() == 1) {
				double epsilon = lexical_cast<double>(arguments[0]);
				f.epsilon = epsilon;
				f.epsDecrease = NAN;
				f.epsIncrease = NAN;
				cout << "Decay : BoldDriver( " << epsilon <<  " ) " << endl;
				f.decayName = "BoldDriver";
				f.decayArgs = arguments;
				return;
			} else if (arguments.size() == 3) {
				double epsilon = lexical_cast<double>(arguments[0]);
				double epsDecrease = lexical_cast<double>(arguments[1]);
				double epsIncrease = lexical_cast<double>(arguments[2]);
				f.epsilon = epsilon;
				f.epsDecrease = epsDecrease;
				f.epsIncrease = epsIncrease;
				cout << "Decay : Auto( " << epsilon << ", " << epsDecrease << ", " << epsIncrease << " ) " << endl;
				f.decayName = "BoldDriver";
				f.decayArgs = arguments;
				return;
			}else {
				goto error;
			}
		} else if (name.compare("Sequential") == 0) {
			if (arguments.size() == 1) {
				double epsilon = lexical_cast<double>(arguments[0]);
				f.epsilon = epsilon;
				f.alpha = NAN;
				f.A = NAN;
				cout << "Decay : Sequential( " << epsilon <<  " ) " << endl;
				f.decayName = "Sequential";
				f.decayArgs = arguments;
				return;
			} else if (arguments.size() == 3) {
				double epsilon = lexical_cast<double>(arguments[0]);
				double alpha = lexical_cast<double>(arguments[1]);
				double A = lexical_cast<double>(arguments[2]);
				f.epsilon = epsilon;
				f.alpha = alpha;
				f.A = A;
				cout << "Decay : Sequential( " << epsilon << ", " << alpha << ", " << A << " ) " << endl;
				f.decayName = "Sequential";
				f.decayArgs = arguments;
				return;
			}else {
				goto error;
			}
		}
	} catch(bad_lexical_cast &) {
	}

	error:
	cerr << "Error in argument '" << argName << "': Unknown or invalid loss '";
	printArg(cerr, name, arguments);
	cerr << "'" << endl;
	cerr << "Valid choices are: " << endl;
	cerr << "    Const(<epsilon>)" << endl;
	cerr << "    Auto(<epsilon>,<tries>,<input-sample-matrix>)" << endl;
	cerr << "    BoldDriver(<epsilon>)" << endl;
	cerr << "    BoldDriver(<epsilon>,<epsDecrease>,<epsIncrease>)" << endl;
	cerr << "    Sequential(<epsilon>)" << endl;
	cerr << "    Sequential(<epsilon>,<alpha>,<A>)" << endl;
	exit(1);
}

}

#endif
