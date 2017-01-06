#include <mpi2/mpi2.h>
#include <mf/mf.h>
#include <mf/matrix/io/generateDistributedMatrix.h>

namespace mf {

using namespace mpi2;

namespace detail {

// task registration
template<typename Types>
void registerMatrixTasksFor() {
	registerTask<CreateMatrixTask<typename Types::Head> >();
	registerTask<BlockAndLoadMatrixTask<typename Types::Head> >();
	registerTask<ReadDistributedMatrixTask<typename Types::Head> >();
	registerTask<WriteDistributedMatrixTask<typename Types::Head> >();

	registerTask<UnblockTask<typename Types::Head> >();
	registerTask<SumTask<typename Types::Head> >();
	registerTask<L1Task<typename Types::Head> >();
	registerTask<L2Task<typename Types::Head> >();

	registerTask<CrossprodTask<typename Types::Head> >();
	registerTask<TCrossprodTask<typename Types::Head> >();

	registerMatrixTasksFor<typename Types::Tail>();
};

template<>
void registerMatrixTasksFor<Nil>() {
};

template<typename Types>
void registerSparseMatrixTasksFor() {
	registerTask<NnzTask<typename Types::Head> >();

	registerSparseMatrixTasksFor<typename Types::Tail>();
};

template<>
void registerSparseMatrixTasksFor<Nil>() {
};

template<typename Types>
void registerDenseMatrixTasksFor() {
	registerTask<ProjectTask<typename Types::Head> >();
	registerTask<MultTask<typename Types::Head> >();
	registerTask<DivTask<typename Types::Head> >();

	registerDenseMatrixTasksFor<typename Types::Tail>();
};

template<>
void registerDenseMatrixTasksFor<Nil>() {
};

void registerMatrixTasks() {
	registerMatrixTasksFor<MatrixTypes>();
	registerSparseMatrixTasksFor<SparseMatrixTypes>();
	registerDenseMatrixTasksFor<DenseMatrixTypes>();
	registerTask<NzslTask<SparseMatrix, DenseMatrix, DenseMatrixCM> >();
	registerTask<SlDataTask<SparseMatrix, DenseMatrix, DenseMatrixCM> >();
	registerTask<KlTask<SparseMatrix, DenseMatrix, DenseMatrixCM> >();
	registerTask<GklDataTask<SparseMatrix, DenseMatrix, DenseMatrixCM> >();
	registerTask<Sums1Task<DenseMatrixCM> >();
	registerTask<Sums2Task<DenseMatrix> >();
	registerTask<SquaredSums1Task<DenseMatrixCM> >();
	registerTask<SquaredSums2Task<DenseMatrix> >();
	registerTask<Div1Task<DenseMatrixCM, boost::numeric::ublas::vector<double> > >();
	registerTask<Mult1Task<DenseMatrixCM, boost::numeric::ublas::vector<double> > >();
	registerTask<Mult2Task<DenseMatrix, boost::numeric::ublas::vector<double> > >();
	registerTask<GklApTaskW>();
	registerTask<NzslApTaskWThreads>();
	registerTask<SlDataApTaskW>();
 	dlee01GklRegisterTasks();
	dalsRegisterTasks();
 	dgnmfRegisterTasks();
	registerTask<Nnz12Task<SparseMatrix> >();
	registerTask<Nzl2LossTask>();
	mpi2::registerTask<mf::detail::AsgdInitTask>();
	mpi2::registerTask<mf::detail::AsgdShuffleTask>();
	mpi2::registerTask<mf::detail::AsgdDestroyTask>();

	registerTask<mf::detail::GenerateRandomFactorsTask<DenseMatrix> >();
	registerTask<mf::detail::GenerateRandomFactorsTask<DenseMatrixCM> >();
	registerTask<mf::detail::GenerateRandomDataMatrixTask<SparseMatrix> >();
	registerTask<mf::detail::GenerateRandomDataMatrixTask<SparseMatrixCM> >();

	// added by me
	registerTask<Nzl2SquaredSumsTask>();

	// ASGD tasks (TODO: generate automatically)
	registerTask<mf::detail::AsgdTask<UpdateNzsl,RegularizeNone> >();
	registerTask<mf::detail::AsgdTask<UpdateTruncate<UpdateNzsl>,RegularizeNoneTruncate> >();
	registerTask<mf::detail::AsgdTask<UpdateNzslL2,RegularizeNone> >();
	registerTask<mf::detail::AsgdTask<UpdateTruncate<UpdateNzslL2>,RegularizeNoneTruncate> >();
	registerTask<mf::detail::AsgdTask<UpdateNzslNzl2,RegularizeNone> >();
	registerTask<mf::detail::AsgdTask<UpdateTruncate<UpdateNzslNzl2>,RegularizeNoneTruncate> >();

	registerTask<mf::detail::DsgdPpTask<UpdateNzsl,RegularizeNone> >();
	registerTask<mf::detail::DsgdPpTask<UpdateTruncate<UpdateNzsl>,RegularizeNoneTruncate> >();
	registerTask<mf::detail::DsgdPpTask<UpdateNzslL2,RegularizeNone> >();
	registerTask<mf::detail::DsgdPpTask<UpdateTruncate<UpdateNzslL2>,RegularizeNoneTruncate> >();
	registerTask<mf::detail::DsgdPpTask<UpdateNzslNzl2,RegularizeNone> >();
	registerTask<mf::detail::DsgdPpTask<UpdateTruncate<UpdateNzslNzl2>,RegularizeNoneTruncate> >();

	registerTask<BiasedNzslTask<SparseMatrix, DenseMatrix, DenseMatrixCM> >();
	registerTask<BiasedNzl2FactorsTask>();
	registerTask<BiasedNzl2BiasTask>();
};

} } // namespace mf::detail
