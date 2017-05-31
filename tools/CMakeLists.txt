link_libraries(${Boost_LIBRARIES} ${RG_LIBRARIES} mpi2 mf log4cxx boost_filesystem boost_system)
file(COPY "${MF_SOURCE_DIR}/log4j.properties" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")
file(COPY "experiments" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")
file(COPY "mfplot.R" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

add_executable(mfdap mfdap.cc)
add_executable(mfasgd mfasgd.cc)
add_executable(mfdsgdpp mfdsgdpp.cc)
add_executable(mfdsgd mfdsgd.cc detail/mfdsgd-generated_impl.cc detail/mfdsgd-generated_impl1.cc detail/mfdsgd-generated_impl2.cc detail/mfdsgd-generated_impl3.cc detail/mfdsgd-generated_impl4.cc detail/mfdsgd-generated_impl5.cc detail/mfdsgd-generated_impl6.cc detail/mfdsgd-generated_impl7.cc detail/mfdsgd-generated_impl8.cc )


add_executable(mfconvert mfconvert.cc)
add_executable(mfdblock mfdblock.cc)
add_executable(mfgenerate mfgenerate.cc)
add_executable(mfsgd mfsgd.cc)

add_executable(mfsample mfsample.cc)
add_executable(mfproject mfproject.cc)
add_executable(mfprepare mfprepare.cc)
add_executable(mfcreateRandomMatrixFile mfcreateRandomMatrixFile.cc)

add_executable(psgdL2NoLock psgdL2NoLock.cc)
add_executable(psgdL2Lock psgdL2Lock.cc)
add_executable(psgdNZL2NoLock psgdNZL2NoLock.cc)
add_executable(psgdNZL2Lock psgdNZL2Lock.cc)
add_executable(stratified-psgd stratified-psgd.cc)

add_executable(mfcreateInitialFactors mfcreateInitialFactors.cc)
add_executable(generateSyntheticData generateSyntheticData.cc)
add_executable(averageOutMatrix averageOutMatrix.cc)

