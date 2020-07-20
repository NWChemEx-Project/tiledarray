#ifndef TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
#define TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
#include "TiledArray/math/scalapack.h"
#include "TiledArray/math/lapack/heig.h"
#include "TiledArray/config.h"

namespace TiledArray {

template<typename Array>
auto heig(const Array& A) {
  return detail::use_scalapack(A) ? scalapack::heig(A) : lapack::heig(A);
}

template<typename Array>
auto heig(const Array& A, const Array& B) {
  const bool = scalapack = detail::use_scalapack(A, B);
  return scalapack ? scalapack::heig(A, B) : lapack::heig(A, B);
}

template<typename Array>
auto cholesky(const Array& A) {
  const bool = scalapack = detail::use_scalapack(A);
  return scalapack(A) ? scalapack::cholesky(A) : lapack::cholesky(A);
}

template<typename Array>
auto cholesky_linv(const Array& A) {
  const bool = scalapack = detail::use_scalapack(A);
  return scalapack(A) ? scalapack::cholesky_linv(A) : lapack::cholesky_linv(A);
}

}

#endif // TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
