#ifndef MPI2_LOGGER_H
#define MPI2_LOGGER_H

#include <log4cxx/logger.h>
#include <log4cxx/mdc.h>

#include <mpi2/task.h>
#include <util/io.h>
#include <util/evaluation/timeval.h>

namespace mpi2 {

namespace detail {
extern log4cxx::LoggerPtr logger;
extern log4cxx::LoggerPtr eventLogger;

inline void logEvent(const std::string& type, const std::string& event){
	if (detail::eventLogger->isInfoEnabled()) {
		log4cxx::MDC mdc1("eventtype", type);
		timeval t;
		gettimeofday(&t, 0);
		uint64_t nanos = rg::detail::timevalToNanos(t);
		log4cxx::MDC mdc2("nanotime", rg::paste(nanos));
		LOG4CXX_INFO(detail::eventLogger, event);
	}
}

}

inline void logBeginEvent(const std::string& event){
	detail::logEvent("+", event);
}

inline void logEndEvent(const std::string& event){
	detail::logEvent("-", event);
}

inline void logEvent(const std::string& location, const std::string& event){
	detail::logEvent("=", event);
}

}

#endif
