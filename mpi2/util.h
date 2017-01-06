#ifndef MPI2_UTIL_H
#define MPI2_UTIL_H

namespace mpi2 {

typedef unsigned long PointerIntType;

template<typename T>
PointerIntType pointerToInt(T* p) {
	return (PointerIntType)p;
}

template<typename T>
T* intToPointer(PointerIntType i) {
	return (T*)i;
}


/** Splits the numbers [0,n-1] into p partitions. Returns a vector x of length p+1 such that
 * partition i is given by [p[i], p[i+1]).
 *
 * @param n
 * @param p
 * @return
 */
template<typename I>
inline std::vector<I> split(I n, int p) {
	std::vector<I> result(p+1, (I)0);
	I np = n / p;
	I mp = n % p;
	I pos = 0;
	for (int i=0; i<=p; i++) {
		result[i] = pos;
		pos += np;
		if (mp > 0) {
			pos++;
			mp--;
		}
	}
	return result;
}

}

#endif
