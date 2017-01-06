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
#ifndef SGD_TRACE_H
#define SGD_TRACE_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <boost/foreach.hpp>

#include <mf/types.h>

namespace mf {
/** Describes an entry for the trace in terms of (1) the current epoch (and the time for completing this epoch),
 * (2) the current iteration,  (3) the loss (and the time for computing the loss),
 * (4) the test loss (and the time for computing the test loss)
*/
struct TraceEntry {
	TraceEntry(double loss, double timeLoss, double testLoss=NAN,double timeTestLoss=0.0)
	: epoch(0), iteration(0), loss(loss), timeEpoch(0), timeLoss(timeLoss),
          timeTestLoss(timeTestLoss), testLoss(testLoss) {
        }

	TraceEntry(mf_size_type epoch, mf_size_type iteration, double loss, double timeLoss,double timeEpoch,
			double testLoss=NAN, double timeTestLoss=0.0)
	: epoch(epoch), iteration(iteration), loss(loss), timeEpoch(timeEpoch), timeLoss(timeLoss), 
        timeTestLoss(timeTestLoss), testLoss(testLoss) {
        }

	virtual ~TraceEntry() { };

	/** Passes the basic information of the entry into a stream
	 * @param out the stream
	 */
	void basicToRstream(std::stringstream& out){
		std::string sep = ", ";
		out << "list(";
		out << "it=" << iteration << ", ";
		out << "epoch=" << epoch << ", ";
		out << "loss=" << loss  << ", ";
		out << "loss.test=";
		if (isnan(testLoss)) {
			out << "NA"<< ", ";
		}
		else {
			out << testLoss<< ", ";
		}
	}
	/** Passes the additional information of the entry
	 * (the info that is different between parent and derived struct) into a stream
	 * @param out the stream
	 * @param elapsedTime the time elapsed so far (it needs to be passed by reference)
	 */
	virtual void moreToRstream(std::stringstream& out,double& elapsedTime){

		elapsedTime += (timeEpoch + timeLoss+timeTestLoss)/1E9;
		out <<"time=list(elapsed=" << elapsedTime<<", epoch="<<timeEpoch/1E9
			<<", loss="<< timeLoss/1E9<<", loss.test="<<timeTestLoss/1E9<<") ";

	}

	mf_size_type epoch;
	mf_size_type iteration;
	double loss;
	double timeEpoch;
	double timeLoss;
	double timeTestLoss;
	double testLoss;
};
/** Describes an entry for the trace for the SGD algorithms. It is a derived structure of TraceEntry.
 * Additionally, the entry contains (1) the current step (eps) and (2) the time to find this step
*/
struct SgdTraceEntry: public  TraceEntry{
	SgdTraceEntry(double loss, double timeLoss, double testLoss=NAN, double timeTestLoss=0.0)
	: TraceEntry( loss,  timeLoss, testLoss, timeTestLoss),
	  eps(NAN), timeEps(0) {
        }

	SgdTraceEntry(mf_size_type epoch, mf_size_type iteration, double loss, double eps,
			double timeEps, double timeEpoch, double timeLoss, double testLoss=NAN, double timeTestLoss=0.0)
	: TraceEntry(epoch, iteration, loss,timeLoss,  timeEpoch,  testLoss, timeTestLoss),
	  eps(eps), timeEps(timeEps){
	}

	/** Passes the additional information of the entry
	 * (the info that is different between parent and derived struct) into a stream
	 * @param out the stream
	 * @param elapsedTime the time elapsed so far (it needs to be passed by reference)
	 */
	virtual void moreToRstream(std::stringstream& out,double& elapsedTime){
		elapsedTime += (timeEps + timeEpoch + timeLoss+timeTestLoss)/1E9;
		out << "time=list(elapsed=" << elapsedTime <<", eps="<<timeEps/1E9<<", epoch="<<timeEpoch/1E9
				<<", loss="<< timeLoss/1E9<<", loss.test="<<timeTestLoss/1E9<<"), ";

		out<<" eps=";
		if (isnan(eps)) {
			out << "NA";
		} else {
			out << eps;
		}

	}
	double eps; // learning rate
	double timeEps;
};
/** Describes an entry for the trace for the ALS algorithm.
 * 	It is a derived struct of TraceEntry. Additionally, the entry contains:
 * 	(1) the rescaling factor
 * 	(2) the time for rescaling
 * 	if rescale = ALS_NONE rescale=NULL
 * 	if rescale = ALS_SIMPLE rescale: a vector with a single value inside
 * 	if rescale = ALS_OPTIMAL rescale: a vector with the optimal rescale values inside
*/
struct AlsTraceEntry: public  TraceEntry{
	AlsTraceEntry(double loss, double timeLoss, double testLoss=NAN,double timeTestLoss=0.0)
	: TraceEntry(loss,  timeLoss, testLoss, timeTestLoss),
	  timeRescale(0) {
        }

	AlsTraceEntry(mf_size_type epoch, mf_size_type iteration, double loss, double timeLoss,
			boost::numeric::ublas::vector<double> rescale,	double timeRescale,
			double timeEpoch, double testLoss=NAN, double timeTestLoss=0.0)
	: TraceEntry(epoch, iteration, loss,timeLoss,  timeEpoch,  testLoss, timeTestLoss),
          rescale(rescale), timeRescale(timeRescale) {
        }

	/** Passes the additional information of the entry
	 * (the info that is different between parent and derived struct) into a stream
	 * @param out the stream
	 * @param elapsedTime the time elapsed so far (it needs to be passed by reference)
	 */
	virtual void moreToRstream(std::stringstream& out,double& elapsedTime){

		elapsedTime += (timeRescale + timeEpoch + timeLoss+timeTestLoss)/1E9;
		out << "time=list(elapsed=" << elapsedTime <<", timeRescale="<<timeRescale/1E9<<", epoch="<<timeEpoch/1E9
				<<", loss="<< timeLoss/1E9<<", loss.test="<<timeTestLoss/1E9<<"), ";

		if (epoch % 2 == 0){
			out<<"update="<<"\"W\" , ";
		}
		else{
			out<<"update="<<"\"H\" , ";
		}

		out<<"rescale=";
		if (rescale.empty()) {
			out << "NA";
		} else {
			out<<"list(";
			std::string sep="";
			for (unsigned i=0; i<rescale.size(); i++){
				out<<sep;
				out<<rescale[i];
				sep=", ";
			}
			out<<")";
		}
	}


	boost::numeric::ublas::vector<double> rescale;
	double timeRescale;
};
/** Describes the trace in terms of (1) a vector of pointers to trace entries,
 * 	(2) a map for additional fields (describing the specific experiment) of type double,
 * 	(3) a map for additional fields (describing the specific experiment) of type string
*/
struct Trace {
	/** Adds a pointer to an entry to this trace
	 * @param entry the pointer to the entry
	 */
	void add(TraceEntry* entry) {
		trace.push_back(entry);
	}
	void clear() {
		trace.clear();
	}
	/** Adds an additional information fied to this trace
	* @param fieldName the name of the field
	* @param value the value of the field
	*/
	void addField(const std::string& fieldName, double value){
		doubleFields[fieldName]=value;
	}
	/** Adds an additional information field to this trace
	* @param fieldName the name of the field
	* @param value the value of the field
	*/
	void addField(const std::string& fieldName,const std::string& value){
		stringFields[fieldName]=value;
	}
	/** passes the  additional information fields to a stream
	* @param out the stream
	*/
	void writeInfo(std::stringstream& out){
		std::string sep = "";
		out << ", info=list(";
		//first the doubles
			std::map<std::string,double>::iterator it1;
			for ( it1=doubleFields.begin() ; it1 != doubleFields.end(); it1++ ){
				out << sep;
				out << (*it1).first << " = " << (*it1).second ;
				sep=", ";
			}
		//then the strings
		std::map<std::string,std::string>::iterator it2;
		sep=(doubleFields.empty()?"":", ");

			for ( it2=stringFields.begin() ; it2 != stringFields.end(); it2++ ){
				out << sep;
				out << (*it2).first << " = \"" << (*it2).second << "\"";
				sep=", ";
			}
		out<<")"; // close the info list
	}
	/** passes the main information of this trace (info of trace entries) to a stream
	* @param out the stream
	*/
	void toRstream(std::stringstream& out, const std::string& varname = "trace") {
		std::string sep = "";
		out << varname << " <- list(";
		out << "trace=list(";
		double elapsedTime = 0;
		for (unsigned i=0; i<trace.size(); i++){
			out << sep << std::endl;
			trace[i]->basicToRstream(out);
			trace[i]->moreToRstream(out,elapsedTime);
			out << ")";//close the entry info
			sep = ", ";
		}
		out << std::endl << ")";//close trace

		if (!doubleFields.empty()||!stringFields.empty()){
			writeInfo(out);
		}
		out << ");";// close variable
	}
	/** writes the information of this trace to an R-file
	* @param file the R-file name
	* @param varname the name of the R-variable
	* returns a string containing the same information as the file
	*/
	std::string toRfile(const std::string& file, const std::string& varname = "trace") {
		std::stringstream ss;
		toRstream(ss, varname);
		std::ofstream out(file.c_str());
		out<<ss.str();
		out.close();
		return ss.str();
	}

	~Trace() {
		for (unsigned i=0; i<trace.size(); i++){
			delete trace[i];
		}
	}

	std::vector<TraceEntry*> trace;
	std::map<std::string,double> doubleFields;
	std::map<std::string,std::string> stringFields;

};

}

#endif
