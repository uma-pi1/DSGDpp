
# define source files
set(libmpi2_SRCS
	env_impl.cc
	init_impl.cc
	logger_impl.cc
	registry_impl.cc
	task-manager_impl.cc
)

# define main header files
set(libmpi2_HDRS
	channel.h
	channel_impl.h
	env.h
	env_fwd.h
	env_impl.h
	marshal.h
	init.h
	mpi2.h	
	logger.h
	registry.h
	seeding.h
	task.h
	task-manager.h
	types.h
	uninitialized.h
	util.h
)

# combine to all headers 
set(libmpi2_all_HDRS
	${libmpi2_HDRS}
)

# add library targets
add_library(mpi2 SHARED ${libmpi2_all_HDRS} ${libmpi2_SRCS}) 

# link libraries
target_link_libraries(mpi2 ${Boost_LIBRARIES} log4cxx)

# install library and main headers
install(TARGETS mpi2 DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${libmpi2_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/mpi2)

