/** \file
 *
 * Provides a way to instantiate classes without correctly initializing them. Such instantiations
 * are needed, for example, when a class is deserialized: first an uninitialized instance is
 * created, then the member variables are set. Use this instead of the empty constructor to
 * clearly mark that the constructor is dangerous to use.
 *
 * Example usage:
 * \code
 * class Test {
 * public:
 *   Test(...) { ... };                                 // default constructor
 *   Test(mpi2::SerializationConstructor _) { ... };    // the special constructor w/o correct initialization
 *   ...
 * }
 * MPI2_SERIALIZATION_CONSTRUCTOR(Test)                 // use the special constructor for deserialization
 * ...
 * Test(mpi2::UNINITALIZED);                            // call the constructor (dangerous!)
 * \endcode
 */

#ifndef LIBMPI2_UNINITIALZED
#define LIBMPI2_UNINITIALZED

#include <boost/serialization/serialization.hpp>

namespace mpi2 {

/** Marks constructors that should only be used for serialization (use as the only argument).
 * See documentation of uninitialized.h. */
enum SerializationConstructor { UNINITIALIZED };


/** The default constructor of a class */
template<class T>
struct Mpi2Constructor {
	static inline T* construct() { return ::new T(); }
};

}

/** Registers a serialization constructor of a class for use during
 * deserialization. Use in top namespace only. */
#define MPI2_SERIALIZATION_CONSTRUCTOR(Clazz)                                        \
namespace boost { namespace serialization {                                          \
template<class Archive>                                                              \
void load_construct_data(Archive& ar, Clazz* a, unsigned int const file_version) {   \
	::new(a) Clazz(mpi2::UNINITIALIZED);                                             \
}                                                                                    \
} }                                                                                  \
namespace mpi2 {                                                                     \
template<>                                                                           \
struct Mpi2Constructor<Clazz> {                                                      \
	static inline Clazz* construct() {                                               \
  	  return ::new Clazz(mpi2::UNINITIALIZED);                                       \
   }                                                                                 \
};                                                                                   \
}

/** Registers a serialization constructor of a class with one template parameter
 * for use during deserialization. Use in top namespace only. */
#define MPI2_SERIALIZATION_CONSTRUCTOR1(Clazz)                                          \
namespace boost { namespace serialization {                                             \
template<class Archive, class T1>                                                       \
void load_construct_data(Archive& ar, Clazz<T1>* a, unsigned int const file_version) {  \
	::new(a) Clazz<T1>(mpi2::UNINITIALIZED);                                            \
}                                                                                       \
} }                                                                                     \
namespace mpi2 {                                                                        \
template<typename T1>                                                                   \
struct Mpi2Constructor<Clazz<T1> > {                                                    \
	static inline Clazz<T1>* construct() {                                              \
	  return ::new Clazz<T1>(mpi2::UNINITIALIZED);                                      \
    }                                                                                   \
};                                                                                      \
}

/** Registers a serialization constructor of a class with two template parameters
 * for use during deserialization. Use in top namespace only. */
#define MPI2_SERIALIZATION_CONSTRUCTOR2(Clazz)                                             \
namespace boost { namespace serialization {                                                \
template<class Archive, class T1, class T2>                                                \
void load_construct_data(Archive& ar, Clazz<T1,T2>* a, unsigned int const file_version) {  \
	::new(a) Clazz<T1,T2>(mpi2::UNINITIALIZED);                                            \
}                                                                                          \
} }                                                                                        \
namespace mpi2 {                                                                           \
template<typename T1, typename T2>                                                         \
struct Mpi2Constructor<Clazz<T1, T2> > {                                                   \
	static inline Clazz<T1,T2>* construct() {                                              \
	  return ::new Clazz<T1,T2>(mpi2::UNINITIALIZED);                                      \
    }                                                                                      \
};                                                                                         \
}

/** Registers a serialization constructor of a class with three template parameters
 * for use during deserialization. Use in top namespace only. */
#define MPI2_SERIALIZATION_CONSTRUCTOR3(Clazz)                                             \
namespace boost { namespace serialization {                                                \
template<class Archive, class T1, class T2, class T3>                                      \
void load_construct_data(Archive& ar, Clazz<T1,T2,T3>* a, unsigned int const file_version) {  \
	::new(a) Clazz<T1,T2,T3>(mpi2::UNINITIALIZED);                                         \
}                                                                                          \
} }                                                                                        \
namespace mpi2 {                                                                           \
template<typename T1, typename T2, typename T3>                                            \
struct Mpi2Constructor<Clazz<T1, T2, T3> > {                                               \
	static inline Clazz<T1,T2,T3>* construct() {                                           \
	  return ::new Clazz<T1,T2,T3>(mpi2::UNINITIALIZED);                                   \
    }                                                                                      \
};                                                                                         \
}

#endif
