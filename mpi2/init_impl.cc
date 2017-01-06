#include <log4cxx/mdc.h>

#include <mpi2/env.h>
#include <mpi2/init.h>
#include <mpi2/logger.h>
#include <mpi2/task-manager.h>

#include <util/io.h>

namespace mpi2 {

boost::mpi::communicator& mpi2init(int& argc, char**& argv) {
	int threadLevel = MPI::Init_thread(argc, argv, MPI_THREAD_MULTIPLE);
	if (threadLevel != MPI_THREAD_MULTIPLE) {
		std::cerr << "Fatal error: Thread level MPI_THREAD_MULTIPLE not supported.";
		exit(1);
	}

	// initialize local task manager
	if (TaskManager::theTaskManager_ != NULL) {
		RG_THROW(rg::IllegalStateException, "mpi2 already initialized");
	}
	TaskManager::theTaskManager_ = new TaskManager(0); // default tag of TM is 0
	boost::mpi::communicator& world = TaskManager::getInstance().world();
	if (world.rank() == 0) {
		LOG4CXX_INFO(detail::logger, "Initialized mpi2 on " << world.size() << " rank(s)");
	}

	// initialize env on this node
	initEnv(world.rank());
	detail::registerEnvTasks();

	// initialize event logger
	char hostname[128];
	size_t len = 126;
	gethostname(hostname, len);
	log4cxx::MDC::put("hostname", std::string(hostname));
	log4cxx::MDC::put("rank", rg::paste(world.rank()));
	log4cxx::MDC::put("task", "main");

	return world;
}

namespace detail {
	boost::thread* tmThread = NULL;
}
void mpi2start(unsigned pollDelay) {
	TaskManager& tm = TaskManager::getInstance();
	tm.setPollDelay(pollDelay);
	boost::mpi::communicator& world = tm.world();
	if ( tm.parallelMode() ) {
		LOG4CXX_INFO(detail::logger, "Starting task managers (parallel mode)...");
		LOG4CXX_INFO(detail::logger, "Started task manager at rank " << tm.world().rank());
		// parallel mode: nothing needs to be done
	} else {
		// distributed mode: spawn thread for task manager
		if (world.rank() == 0) {
			LOG4CXX_INFO(detail::logger, "Starting task managers (distributed mode, polling delay: " << pollDelay << " microseconds)...");
			detail::tmThread = new boost::thread(boost::bind(&TaskManager::run,
					&tm));
		} else {
			tm.run();
		}
	}
}

void mpi2stop() {
	boost::mpi::communicator& world = TaskManager::getInstance().world();
	if (world.rank() == 0) {
		TaskManager& tm = TaskManager::getInstance();
		LOG4CXX_INFO(detail::logger, "Stopping task managers...");

		if ( tm.parallelMode() ) {
			tm.shutdown();
			LOG4CXX_INFO(detail::logger, "Shutdown task manager at rank " << tm.world().rank());
		} else {
			boost::mpi::request send_reqs[world.size()];
			boost::mpi::request recv_reqs[world.size()];
			int responseTag = tm.unusedTag();
			TaskManager::TaskRequest req(responseTag, TaskManager::TASK_ID_QUIT, responseTag);
			for (int p=0; p<world.size(); p++) {
				send_reqs[p] = world.isend(p, tm.tag(), req);
				recv_reqs[p] = world.irecv(p, responseTag);
			}
			boost::mpi::wait_all(recv_reqs, recv_reqs + world.size());
			detail::tmThread->join();
			delete detail::tmThread;
			detail::tmThread = NULL;
		}
	}
}

// finalize mpi
void mpi2finalize() {
	boost::mpi::communicator& world = TaskManager::getInstance().world();
	if (world.rank() == 0) {
		LOG4CXX_INFO(detail::logger, "Shutting down mpi2...");
	}
	MPI::Finalize();
}

} // namespace mpi2
