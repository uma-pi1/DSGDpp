#include <mpi2/logger.h>

namespace mpi2 { namespace detail {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("mpi2"));
log4cxx::LoggerPtr eventLogger(log4cxx::Logger::getLogger("evnt"));
} }


