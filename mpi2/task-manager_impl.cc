#include <mpi.h>

#include <boost/thread/thread.hpp>

#include <util/io.h>

#include <mpi2/env.h>
#include <mpi2/logger.h>
#include <mpi2/registry.h>
#include <mpi2/task.h>


namespace mpi2 {

const std::string TaskManager::TASK_ID_QUIT = "__mpi2/TaskManager/quit";
TaskManager* TaskManager::theTaskManager_ = NULL;

int TaskManager::unusedTag() {
	boost::mutex::scoped_lock lock(nextTagMutex_);
	if (nextTag_+1 > MPI_TAG_UB) {
		RG_THROW(rg::IllegalStateException, "run out of tags");
	}
	return nextTag_++;
}

void TaskManager::unusedTags(int n, std::vector<int>& tags) {
	boost::mutex::scoped_lock lock(nextTagMutex_);
	tags.resize(n);
	for (int i=0; i<n; i++) {
		if (nextTag_+1 > MPI_TAG_UB) {
			RG_THROW(rg::IllegalStateException, "run out of tags");
		}
		tags[i] = nextTag_++;
	}
}

Channel TaskManager::spawn(int rank, TaskId task) {
	BOOST_ASSERT( rank>=0 && rank<world_.size() );

	boost::mutex::scoped_lock lock(spawnMutex_); // lock needed here; not sure why right now
	int responseTag = unusedTag();
	int sourceTag = unusedTag();

	TaskRequest taskRequest(responseTag, task, sourceTag);
	TaskResponse taskResponse;
	if ( rank == world_.rank() ) { // no communication needed
		taskResponse.taskTags = schedule(taskRequest);
	} else {
		world_.send(rank, tag(), taskRequest);
		world_.recv(rank, responseTag, taskResponse);
	}

	return Channel(world_, Endpoint(world_.rank(), sourceTag),
			Endpoint(rank, taskResponse.taskTags[0]));
}

void TaskManager::spawn(int rank, TaskId taskId, int n, std::vector<Channel>& channels, bool pairwiseChannels) {
	BOOST_ASSERT( rank>=0 && rank<world_.size() );

	boost::mutex::scoped_lock lock(spawnMutex_); // lock needed here; not sure why right now
	int responseTag = unusedTag();
	std::vector<int> sourceTags(n);
	unusedTags(n, sourceTags);
	TaskRequest taskRequest(responseTag, taskId, sourceTags, pairwiseChannels, n);
	TaskResponse taskResponse;

	if ( rank == world_.rank() ) { // no communication needed
		taskResponse.taskTags = schedule(taskRequest);
	} else {
		world_.send(rank, tag(), taskRequest);

		if (pairwiseChannels) { // TODO: could optimize this case... (no need for communication)
			std::vector<int> tags(n*n);
			world_.recv(rank, responseTag, tags);
			std::vector<Endpoint> endpoints(n*n, UNINITIALIZED);
			for (int from=0; from<n; from++) {
				for (int to=0; to<n; to++) {
					endpoints[from*n+to] = Endpoint(rank, tags[from*n+to]);
				}
			}
			world_.send(rank, tags[0], endpoints);
		}

		world_.recv(rank, responseTag, taskResponse);
	}

	channels.resize(n, UNINITIALIZED);
	for (int i=0; i<n; i++) {
		channels[i] = Channel(world_, Endpoint(world_.rank(), sourceTags[i]),
				Endpoint(rank, taskResponse.taskTags[i]));
	}
}

void TaskManager::spawnAll(TaskId taskId, std::vector<Channel>& channels, bool pairwiseChannels) {
	if ( world_.size() == 1 ) { // no communication needed
		spawn(0, taskId, 1, channels, pairwiseChannels);
		return;
	}

	boost::mutex::scoped_lock lock(spawnMutex_); // lock needed here; not sure why right now
	int responseTag = unusedTag();
	const int m = world_.size();
	int sourceTag = unusedTag();
	boost::mpi::request send_reqs[m];
	boost::mpi::request recv_reqs[m];
	TaskRequest taskRequests[m];
	TaskResponse taskResponses[m];
	for (int rank=0; rank<m; rank++) {
		taskRequests[rank] = TaskRequest(responseTag, taskId, sourceTag, pairwiseChannels, m, rank);
		send_reqs[rank] = world_.isend(rank, tag(), taskRequests[rank]);
	}
	boost::mpi::wait_all(send_reqs, send_reqs + m);

	if (pairwiseChannels) {
		std::vector<std::vector<int> > tags(m); // inner vector has 1 element
		for (int rank=0; rank<m; rank++) { // receive
			world_.recv(rank, responseTag, tags[rank]);
		}
		std::vector<Endpoint> endpoints(m*m, UNINITIALIZED);
		for (int fromRank=0; fromRank<m; fromRank++) {
			for (int toRank=0; toRank<m; toRank++) {
				endpoints[fromRank*m+toRank] = Endpoint(toRank, tags[toRank][0]);
			}
		}
		for (int rank=0; rank<m; rank++) { // send
			send_reqs[rank] = world_.isend(rank, tags[rank][0], endpoints);
		}
		boost::mpi::wait_all(send_reqs, send_reqs+m);
	}

	for (int rank=0; rank<m; rank++) {
		recv_reqs[rank] = world_.irecv(rank, responseTag, taskResponses[rank]);
	}
	boost::mpi::wait_all(recv_reqs, recv_reqs+m);

	channels.resize(m, UNINITIALIZED);
	for (int rank=0; rank<m; rank++) {
		channels[rank] = Channel(world_, Endpoint(world_.rank(), sourceTag),
				Endpoint(rank, taskResponses[rank].taskTags[0]));
	}
}

void TaskManager::spawnAll(TaskId taskId, int n, std::vector<Channel>& channels, bool pairwiseChannels) {
	if ( world_.size() == 1 ) { // no communication needed
		spawn(0, taskId, n, channels, pairwiseChannels);
		return;
	}

	boost::mutex::scoped_lock lock(spawnMutex_); // lock needed here; not sure why right now
	int responseTag = unusedTag();
	const int m = world_.size();
	const int g = m*n;
	boost::mpi::request send_reqs[m];
	boost::mpi::request recv_reqs[m];
	std::vector<int> sourceTags(n);
	unusedTags(n, sourceTags);
	TaskRequest taskRequests[m];
	TaskResponse taskResponses[m];
	for (int rank=0; rank<m; rank++) {
		taskRequests[rank] = TaskRequest(responseTag, taskId, sourceTags, pairwiseChannels, m*n, rank*n);
		send_reqs[rank] = world_.isend(rank, tag(), taskRequests[rank]);
	}
	boost::mpi::wait_all(send_reqs, send_reqs + m);

	if (pairwiseChannels) {
		std::vector<std::vector<int> > tags(m); // inner vector has n*n elements
		for (int rank=0; rank<m; rank++) { // receive
			world_.recv(rank, responseTag, tags[rank]);
		}
		std::vector<Endpoint> endpoints(g*g, UNINITIALIZED);
		for (int fromRank=0; fromRank<m; fromRank++) {
			for (int fromTask=0; fromTask<n; fromTask++) {
				for (int toRank=0; toRank<m; toRank++) {
					for (int toTask=0; toTask<n; toTask++) {
						int from = fromRank*n+fromTask;
						int to = toRank*n+toTask;
						endpoints[from*g+to] = Endpoint(toRank, tags[toRank][fromTask*n+toTask]);
					}
				}
			}
		}
		for (int rank=0; rank<m; rank++) { // send
			send_reqs[rank] = world_.isend(rank, tags[rank][0], endpoints);
		}
		boost::mpi::wait_all(send_reqs, send_reqs+m);
	}


	for (int rank=0; rank<m; rank++) {
		recv_reqs[rank] = world_.irecv(rank, responseTag, taskResponses[rank]);
	}
	boost::mpi::wait_all(recv_reqs, recv_reqs + m);

	channels.resize(m*n, UNINITIALIZED);
	for (int rank=0; rank<m; rank++) {
		for (int task=0; task<n; task++) {
			channels[rank*n+task] = Channel(world_, Endpoint(world_.rank(), sourceTags[task]),
					Endpoint(rank, taskResponses[rank].taskTags[task]));
		}
	}
}


void TaskManager::checkPendingRequests() {
	boost::mutex::scoped_lock lock(pendingRequestsMutex_);
	for (unsigned i = 0; i<pendingRequests_.size(); i++) {
		boost::optional<boost::mpi::status> msg = pendingRequests_[i].test();
		if (msg) {
			if (i < pendingRequests_.size()-1) {
				pendingRequests_[i] = pendingRequests_[pendingRequests_.size()-1];
			}
			pendingRequests_.pop_back();
			i--;
		}
	}
}

void TaskManager::run() {
	LOG4CXX_INFO(detail::logger, "Started task manager at rank "<< world_.rank());

	// main loop
	bool done = false;
	TaskRequest taskRequest;
	boost::optional<boost::mpi::status> msg;
	while (!done) {
		checkPendingRequests();
		boost::mpi::request req = world_.irecv(boost::mpi::any_source, tag_, taskRequest);
		while ( !(msg = req.test()) ) {
			boost::this_thread::sleep(boost::posix_time::microsec(pollDelay_));
			checkPendingRequests();
		}
		if (taskRequest.taskId == TASK_ID_QUIT) {
			done = true;
		} else {
			if (TaskRegistry::getInstance().get(taskRequest.taskId) == NULL) {
				RG_THROW(rg::InvalidArgumentException, rg::paste(
						"unknown task id ", taskRequest.taskId, " received from rank ",
						msg->source(), " at ", world_.rank()));
			}
			schedule(msg->source(), taskRequest);
		}
	}

	// shutting down
	shutdown();
	world_.send(msg->source(), taskRequest.sourceTags[0]);
	LOG4CXX_INFO(detail::logger, "Shutdown task manager at rank " << world_.rank());
}

void TaskManager::shutdown() {
	// waiting for all threads to finish
	threadPool_.wait();
}

class TaskRunner {
public:
	TaskRunner(boost::mpi::communicator& world, int source, int sourceTag, Task task, int taskTag, TaskInfo info)
	: world_(world), source_(source), sourceTag_(sourceTag), task_(task), taskTag_(taskTag), info_(info)
	{
	}

	void operator()() const {
		// initialize event logger
		char hostname[128];
		size_t len = 126;
		gethostname(hostname, len);
		log4cxx::MDC mdc1("hostname", std::string(hostname));
		log4cxx::MDC mdc2("rank", rg::paste(world_.rank()));
		log4cxx::MDC mdc3("task", rg::paste(taskTag_));

		// run task
		task_(Channel(world_, Endpoint(world_.rank(), taskTag_), Endpoint(source_, sourceTag_)), info_);
	}

private:
	boost::mpi::communicator& world_;
	int source_;
	int sourceTag_;
	Task task_;
	int taskTag_;
	TaskInfo info_;
};

void TaskManager::startTask(boost::mpi::communicator& world, int source, int sourceTag, Task task, int taskTag, TaskInfo info) {
	// wait until all scheduled tasks have been spawned
	// this is guaranteed to not block since we make sure that the number of workers is awlays
	// sufficient
	while (!threadPool_.empty()) { };

	// add a worker thread, if needed
	std::size_t size = threadPool_.size();
	std::size_t needSize = threadPool_.active() + 1;
	if (size < needSize) {
		LOG4CXX_DEBUG(detail::logger, "Increased thread pool size to " << needSize << " at rank " << world.rank());
		threadPool_.size_controller().resize(needSize);
	}

	// schedule the task
	threadPool_.schedule( TaskRunner(world, source, sourceTag, task, taskTag, info) );
}

// scheduler (distributed mode)
void TaskManager::schedule(int source, TaskRequest taskRequest) {
	BOOST_ASSERT( !parallelMode() );

	boost::mutex::scoped_lock lock(mutex_);
	Task task = TaskRegistry::getInstance().get(taskRequest.taskId);
	int n = taskRequest.sourceTags.size();
	LOG4CXX_DEBUG(detail::logger, "Spawning " << n << " instances(s) of " << taskRequest.taskId << " at rank " << world_.rank());

	// Send tag information for establishing pairwise channels and receive back a list of
	// all endpoints used for communication.
	//
	// This node sends n*n fresh tags to the spawner (where n is the number of local tasks).
	// Consider two tasks i1 at rank r1 and i2 at rank r2. If i1 wants to send a message to
	// i2, it sends to (r2, r2tags(i1,i2)). The task manager sends back these g*g endpoints
	// (task x task; g is the number of total tasks)
	int g = taskRequest.groupSize;
	std::vector<Endpoint> pairwiseEndpoints(g*g, UNINITIALIZED);
	if (taskRequest.pairwiseChannels) {
		std::vector<int> pairwiseTags(n*n);
		unusedTags(n*n, pairwiseTags);
		world_.send(source, taskRequest.responseTag, pairwiseTags);
		world_.recv(source, pairwiseTags[0], pairwiseEndpoints);
	}

	std::vector<int> taskTags(n);
	unusedTags(n, taskTags);
	for (int t=0; t<n; t++) {
		TaskInfo info(taskRequest.groupSize, taskRequest.groupOffset + t);
		if (taskRequest.pairwiseChannels) {
			// create a list of channels using the appropriate endpoints
			int from = taskRequest.groupOffset + t;
			info.pairwiseChannels_.resize(g, UNINITIALIZED);
			for (int to=0; to<g; to++) {
				info.pairwiseChannels_[to] = Channel(world_,
						pairwiseEndpoints[to*g+from],
						pairwiseEndpoints[from*g+to]);
			}
			info.pairwiseChannels_[from].deactivate();
		}
		startTask(world_, source, taskRequest.sourceTags[t],
			task, taskTags[t], info);
	}
	TaskResponse taskResponse(taskTags);
	world_.send(source, taskRequest.responseTag, taskResponse);
}

// scheduler (parallel mode)
std::vector<int> TaskManager::schedule(TaskRequest taskRequest) {
	//BOOST_ASSERT( parallelMode() );
	if (TaskRegistry::getInstance().get(taskRequest.taskId) == NULL) {
		RG_THROW(rg::InvalidArgumentException, rg::paste(
				"unknown task id ", taskRequest.taskId, " received from rank ",
				world_.rank(), " at ", world_.rank()));
	}

	boost::mutex::scoped_lock lock(mutex_);
	Task task = TaskRegistry::getInstance().get(taskRequest.taskId);
	int n = taskRequest.sourceTags.size();
	BOOST_ASSERT( taskRequest.groupOffset == 0);
	BOOST_ASSERT( n == taskRequest.groupSize );
	LOG4CXX_DEBUG(detail::logger, "Spawning " << n << " instance(s) of " << taskRequest.taskId << " at rank " << world_.rank());

	// obtain tags for pairwise communication
	std::vector<int> pairwiseTags;
	if (taskRequest.pairwiseChannels) {
		unusedTags(n*n, pairwiseTags);
	}

	// spawn the threads
	std::vector<int> taskTags(n);
	unusedTags(n, taskTags);
	for (int from=0; from<n; from++) {
		TaskInfo info(taskRequest.groupSize, from);
		if (taskRequest.pairwiseChannels) {
			// create a list of channels using the appropriate endpoints
			info.pairwiseChannels_.resize(n, UNINITIALIZED);
			for (int to=0; to<n; to++) {
				info.pairwiseChannels_[to] = Channel(world_,
						Endpoint(0, pairwiseTags[to*n+from]),
						Endpoint(0, pairwiseTags[from*n+to]));
			}
			info.pairwiseChannels_[from].deactivate();
		}
		startTask(world_, world_.rank(), taskRequest.sourceTags[from],
				task, taskTags[from], info);
	}

	return taskTags;
}


} // namespace mpi2
