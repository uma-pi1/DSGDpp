link_libraries(${Boost_LIBRARIES} ${RG_LIBRARIES} mpi2 log4cxx)
file(COPY "${MF_SOURCE_DIR}/log4j.properties" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

install(FILES  DESTINATION /home/rg)

add_executable(env env.cc)
add_executable(marshal marshal.cc)
add_executable(spawn spawn.cc)
add_executable(pairwise pairwise.cc)
add_executable(pi_task pi_task.cc)
add_executable(udt udt.cc)
add_executable(mpi2test mpi2test.cc)
add_executable(event event.cc)


