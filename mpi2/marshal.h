#ifndef MPI2_GROUP_H
#define MPI2_GROUP_H

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>

namespace mpi2 {

// -- 2 values ------------------------------------------------------------------------------------

namespace detail {
template<typename T1, typename T2>
struct MarshalValues2 {
	MarshalValues2(const T1& v1, const T2& v2) : v1(v1), v2(v2) { }

private:
	const T1& v1; const T2& v2;

	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const {
		ar & v1; ar & v2;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // load not supported
};

template<typename T1, typename T2>
struct UnmarshalValues2 {
	UnmarshalValues2(T1& v1, T2& v2) : v1(v1), v2(v2) { }

private:
	T1& v1;	T2& v2;

	friend class boost::serialization::access;
	template<class Archive>
	void load(Archive & ar, const unsigned int version) {
		ar & v1; ar & v2;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // save not supported
};

}

template<typename T1, typename T2>
detail::MarshalValues2<T1,T2> marshal(const T1& v1, const T2& v2) {
	return detail::MarshalValues2<T1,T2>(v1,v2);
}

template<typename T1, typename T2>
boost::shared_ptr<detail::UnmarshalValues2<T1,T2> > unmarshal(T1& v1, T2& v2) {
	return boost::shared_ptr<detail::UnmarshalValues2<T1,T2> >(new detail::UnmarshalValues2<T1,T2>(v1,v2));
}


// -- 3 values ------------------------------------------------------------------------------------

namespace detail {

template<typename T1, typename T2, typename T3>
struct MarshalValues3 {
	MarshalValues3(const T1& v1, const T2& v2, const T3& v3) : v1(v1), v2(v2), v3(v3) { }

private:
	const T1& v1; const T2& v2; const T3& v3;

	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const {
		ar & v1; ar & v2; ar & v3;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // load not supported
};

template<typename T1, typename T2, typename T3>
struct UnmarshalValues3 {
	UnmarshalValues3(T1& v1, T2& v2, T3& v3) : v1(v1), v2(v2), v3(v3) { }

private:
	T1& v1;	T2& v2; T3& v3;

	friend class boost::serialization::access;
	template<class Archive>
	void load(Archive & ar, const unsigned int version) {
		ar & v1; ar & v2; ar & v3;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // save not supported
};

}

template<typename T1, typename T2, typename T3>
detail::MarshalValues3<T1,T2,T3> marshal(const T1& v1, const T2& v2, const T3& v3) {
	return detail::MarshalValues3<T1,T2,T3>(v1,v2,v3);
}

template<typename T1, typename T2, typename T3>
boost::shared_ptr<detail::UnmarshalValues3<T1,T2,T3> > unmarshal(T1& v1, T2& v2, T3& v3) {
	return boost::shared_ptr<detail::UnmarshalValues3<T1,T2,T3> >(new detail::UnmarshalValues3<T1,T2,T3>(v1,v2,v3));
}


// -- 4 values ------------------------------------------------------------------------------------

namespace detail {

template<typename T1, typename T2, typename T3, typename T4>
struct MarshalValues4 {
	MarshalValues4(const T1& v1, const T2& v2, const T3& v3, const T4& v4) : v1(v1), v2(v2), v3(v3), v4(v4) { }

private:
	const T1& v1; const T2& v2; const T3& v3; const T4& v4;

	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const {
		ar & v1; ar & v2; ar & v3; ar & v4;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // load not supported
};

template<typename T1, typename T2, typename T3, typename T4>
struct UnmarshalValues4 {
	UnmarshalValues4(T1& v1, T2& v2, T3& v3, T4& v4) : v1(v1), v2(v2), v3(v3), v4(v4) { }

private:
	T1& v1;	T2& v2; T3& v3; T3& v4;

	friend class boost::serialization::access;
	template<class Archive>
	void load(Archive & ar, const unsigned int version) {
		ar & v1; ar & v2; ar & v3; ar & v4;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER(); // save not supported
};

}

template<typename T1, typename T2, typename T3, typename T4>
detail::MarshalValues4<T1,T2,T3,T4> marshal(const T1& v1, const T2& v2, const T3& v3, const T4& v4) {
	return detail::MarshalValues4<T1,T2,T3,T4>(v1,v2,v3,v4);
}

template<typename T1, typename T2, typename T3, typename T4>
boost::shared_ptr<detail::UnmarshalValues4<T1,T2,T3,T4> > unmarshal(T1& v1, T2& v2, T3& v3, T4& v4) {
	return boost::shared_ptr<detail::UnmarshalValues4<T1,T2,T3,T4> >(new detail::UnmarshalValues4<T1,T2,T3,T4>(v1,v2,v3,v4));
}



}

#endif
