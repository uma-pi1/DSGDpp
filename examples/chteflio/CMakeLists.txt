
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/examples/chteflio")

link_libraries(${Boost_LIBRARIES} ${RG_LIBRARIES} mpi2 mf log4cxx ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} boost_filesystem boost_system)
file(COPY "${MF_SOURCE_DIR}/log4j.properties" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

add_executable(averageOutMatrix averageOutMatrix.cc)
add_executable(mfcreateInitialFactors mfcreateInitialFactors.cc)
add_executable(generateSyntheticData generateSyntheticData.cc)
add_executable(calculateCSGDblocking calculateCSGDblocking.cc )
add_executable(scaleFactors scaleFactors.cc )
add_executable(createBiasedFactors createBiasedFactors.cc )
add_executable(psgdNZL2BiasedNoLock psgdNZL2BiasedNoLock.cc )