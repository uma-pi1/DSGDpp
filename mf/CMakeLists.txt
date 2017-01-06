# define source files

FILE(GLOB REGISTER_CC "register/register-generated*.cc")
FILE(GLOB REGISTER_H "register/register-generated*.h")

set(libmf_SRCS
	logger_impl.cc
	ap/als_impl.cc
	ap/dals_impl.cc
	ap/lee01-gkl_impl.cc
	ap/dlee01-gkl_impl.cc
	ap/gnmf_impl.cc
	ap/dgnmf_impl.cc
	lapack/lapack_wrapper_impl.cc
	matrix/distribute_impl.cc
	matrix/distributed_matrix_impl.cc
	matrix/op/balance_impl.cc
	matrix/op/project_impl.cc
	matrix/io/descriptor_impl.cc
	register/register_impl.cc
	${REGISTER_CC}	
	init_impl.cc
	matrix/io/mappingDescriptor_impl.cc	
	matrix/io/ioProjected_impl.cc
	matrix/io/randomMatrixDescriptor_impl.cc
)

# define main header files
set(libmf_HDRS
	id.h
	factorization.h
	init.h
	logger.h
	
	loss/loss.h	
	loss/nzsl.h
	loss/nzrmse.h
	loss/l1.h
	loss/l2.h
	loss/nzl2.h	
	loss/sl.h
	loss/sl-data.h
	loss/sl-model.h
	loss/kl.h
	loss/gkl.h
	loss/gkl-data.h
	loss/gkl-model.h
	loss/biased-nzsl.h
	loss/biased-nzl2.h
	
	register/register.h
	${REGISTER_H}	
	mf.h
	trace.h
	types.h
	lapack/blaswrap.h
	lapack/clapack.h
	lapack/lapack_wrapper.h
)

# define additional headers
set(libmf_matrix_HDRS
	matrix/coordinate.h
	matrix/distribute.h
	matrix/distribute_impl.h
	matrix/distributed_matrix.h
	matrix/distributed_matrix_impl.h
	matrix/op/balance.h
	matrix/op/copy.h
	matrix/op/generate.h
	matrix/op/nnz.h
	matrix/op/crossprod.h	
	matrix/op/sum.h
	matrix/op/sums.h
	matrix/op/sumofprod.h
	matrix/op/scale.h
	matrix/op/unblock.h
	matrix/op/project.h
	matrix/op/project_impl.h	
	matrix/op/shuffle.h
	
	matrix/io/format.h
	matrix/io/read.h
	matrix/io/read_impl.h
	matrix/io/write.h
	matrix/io/write_impl.h
	matrix/io/descriptor.h
	matrix/io/descriptor_impl.h	
	matrix/io/randomMatrixDescriptor.h
	matrix/io/load.h
	matrix/io/load_impl.h
	matrix/io/store.h
	matrix/io/store_impl.h
	matrix/io/loadDistributedMatrix.h
	matrix/io/loadDistributedMatrix_impl.h
	matrix/io/mappingDescriptor.h
	matrix/io/ioProjected.h
#	matrix/io/generateDistributedMatrix.h
#	matrix/io/generateDistributedMatrix_impl.h
)

set(libmf_ap_HDRS
	ap/lee01-gkl.h
	ap/dlee01-gkl.h
	ap/gnmf.h
	ap/dgnmf.h
	ap/als.h
	ap/dals.h
	ap/apupdate.h
	ap/aptask.h
	ap/dap-factorization.h
)

set(libmf_sgd_HDRS
	sgd/sgd.h
	sgd/sgd_impl.h
	sgd/asgd.h
	sgd/asgd_impl.h
	sgd/psgd.h
	sgd/psgd_impl.h
	sgd/dsgd.h
	sgd/dsgd_impl.h
	sgd/dsgd-factorization.h
	sgd/dsgdpp.h
	sgd/dsgdpp_impl.h
	sgd/dsgdpp-factorization.h
	sgd/decay/decay.h
	sgd/decay/decay_auto.h
	sgd/decay/decay_bolddriver.h
	sgd/decay/decay_sequential.h
	sgd/decay/decay_constant.h
	sgd/functions/functions.h
	sgd/functions/regularize-none.h
	sgd/functions/regularize-l1.h
	sgd/functions/regularize-l2.h
	sgd/functions/regularize-nzl2.h
	sgd/functions/regularize-sl.h
	sgd/functions/regularize-gkl.h
	sgd/functions/regularize-abs.h
	sgd/functions/regularize-truncate.h
	sgd/functions/regularize-maxnorm.h
	sgd/functions/update-none.h
	sgd/functions/update-nzsl-nzl2.h
	sgd/functions/update-nzsl.h
	sgd/functions/update-sl.h
	sgd/functions/update-nzsl-l2.h
	sgd/functions/update-nzsl-l1.h
	sgd/functions/update-abs.h
	sgd/functions/update-truncate.h
	sgd/functions/update-lock.h
	sgd/functions/update-maxnorm.h	
	sgd/functions/update-biased-nzsl-nzl2.h
)



# combine to all headers 
set(libmf_all_HDRS
	${libmf_HDRS}
	${libmf_matrix_HDRS}
	${libmf_ap_HDRS}
	${libmf_sgd_HDRS}
)

# add library targets
add_library(mf SHARED ${libmf_all_HDRS} ${libmf_SRCS}) 

# link with boost
target_link_libraries(mf mpi2 ${Boost_LIBRARIES} log4cxx blas lapack ${GSL_LIBRARIES})

# install library and main headers
install(TARGETS mf DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${libmf_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/mf)

# install additional headers
install(FILES ${libmf_matrix_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/mf/matrix)
install(FILES ${libmf_sgd_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/mf/sgd)

