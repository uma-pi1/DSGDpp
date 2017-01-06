/** \file
 *
 * Support for compile-time lists of types.
 *
 * This include file defines a compile-time list of types that can be used to run a
 * function for all the types in the list. Used for serialization and task registration. See
 * registerEnvTaskForType in env_impl.cc for an example of usage.
 *
 * Here is an example usage that prints the names of the types in the list (e.g.,
 * call f<Mpi2BuiltinTypes>()).
 * \code
 * template<typename Cons>
 * void f() {
 *     std::cout << Cons::Head::name();
 *     f<typename Cons::Tail>();
 * };
 *
 * template<>
 * void f<Nil>() {
 *    std::cout << "done" << std::endl;
 * };
 * \endcode
 */

#ifndef MPI2_TYPES_H
#define MPI2_TYPES_H

#include <iostream>
#include <vector>

#include <util/exception.h>


namespace mpi2 {

// -- Templates for compile-time lists of types ---------------------------------------------------

/** An empty compile-time list. */
struct Nil {
	/** The head element */
  typedef Nil Head;

  /** The tail */
  typedef Nil Tail;
};

/** Append an element to the front of a compile-time list. */
template<typename Head_, typename Tail_=Nil>
struct Cons {
  typedef Head_ Head;
  typedef Tail_ Tail;
};

/** Concatenates two compile-time lists. */
template<typename List1, typename List2>
struct Concat {
	typedef typename List1::Head Head;
	typedef Concat<typename List1::Tail, List2> Tail;
};

/** Concatenates empty list to the given list.  */
template<typename List2>
struct Concat<Nil,List2> {
	typedef typename List2::Head Head;
	typedef typename List2::Tail Tail;
};

/** Concatenates the given list with an empty list.  */
template<typename List1>
struct Concat<List1,Nil> {
	typedef typename List1::Head Head;
	typedef typename List1::Tail Tail;
};


// -- Templates for type traits -------------------------------------------------------------------

/** A description of a type, including the type itself and the type name. Template specialization
 * may be used to create descriptors of specific types; see macro MPI2_TYPE_TRAITS. */
template<typename T>
struct TypeTraits {
	typedef T Type;
	static inline const char* name() {
		RG_THROW(rg::Exception, std::string("type traits undefined: ") + typeid(T).name());
	}
};

}

/** Registers TypeTraits for the specified type. Only use on global namespace. */
#define MPI2_TYPE_TRAITS(T)                                    \
	namespace mpi2 {                                           \
		template<> struct TypeTraits<T> {                      \
			typedef T Type;                                    \
			static inline const char* name() { return #T; }    \
		};                                                     \
	}

/** Registers TypeTraits for the specified type with 1 template argument.
 * Only use on global namespace. */
#define MPI2_TYPE_TRAITS1(T, T1)                               \
	namespace mpi2 {                                           \
		template<> struct TypeTraits<T<T1> >    {              \
			typedef T<T1> Type;                                \
			static inline const char* name() { return #T "<" #T1 ">"; }    \
		};                                                     \
	}

/** Registers TypeTraits for the specified type with 2 template arguments.
 * Only use on global namespace. */
#define MPI2_TYPE_TRAITS2(T, T1, T2)                           \
	namespace mpi2 {                                           \
		template<> struct TypeTraits<T<T1,T2> > {              \
			typedef T<T1,T2> Type;                             \
			static inline const char* name() { return #T "<" #T1 "," #T2 ">"; }    \
		};                                                     \
	}

/** Registers TypeTraits for the specified type with 3 template arguments.
 * Only use on global namespace. */
#define MPI2_TYPE_TRAITS3(T, T1, T2, T3)                       \
	namespace mpi2 {                                           \
		template<> struct TypeTraits<T<T1,T2,T3> > {           \
			typedef T<T1,T2,T3> Type;                          \
			static inline const char* name() { return #T "<" #T1 "," #T2 "," #T3 ">"; }    \
		};                                                     \
	}

// register some predefined types
MPI2_TYPE_TRAITS(int);
MPI2_TYPE_TRAITS(unsigned);
MPI2_TYPE_TRAITS(long);
MPI2_TYPE_TRAITS(unsigned long);
MPI2_TYPE_TRAITS(double);
MPI2_TYPE_TRAITS(std::string);
MPI2_TYPE_TRAITS1(std::vector, int);
MPI2_TYPE_TRAITS1(std::vector, unsigned);
MPI2_TYPE_TRAITS1(std::vector, long);
MPI2_TYPE_TRAITS1(std::vector, unsigned long);
MPI2_TYPE_TRAITS1(std::vector, double);
MPI2_TYPE_TRAITS1(std::vector, std::string);


// -- Built-in lists of types ---------------------------------------------------------------------

namespace mpi2 {

/** A list of all types built into MPI2. */
typedef
		Cons<int,
		Cons<unsigned,
		Cons<long,
		Cons<unsigned long,
		Cons<double,
		Cons<std::string,
		Cons<std::vector<int>,
		Cons<std::vector<unsigned>,
		Cons<std::vector<long>,
		Cons<std::vector<unsigned long>,
		Cons<std::vector<double>,
		Cons<std::vector<std::string>,
		Nil > > > > > > > > > > > >
Mpi2BuiltinTypes;

}

#endif
