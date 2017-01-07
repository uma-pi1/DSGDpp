link_libraries(${Boost_LIBRARIES} ${RG_LIBRARIES} mpi2 log4cxx boost_filesystem boost_system)
file(COPY "${MF_SOURCE_DIR}/log4j.properties" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

add_subdirectory(mpi2)
add_subdirectory(mf)
#add_subdirectory(rgemulla EXCLUDE_FROM_ALL)
#add_subdirectory(chteflio EXCLUDE_FROM_ALL)
#add_subdirectory(fmakari EXCLUDE_FROM_ALL)

