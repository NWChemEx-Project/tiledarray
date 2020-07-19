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
  return detail::use_scalapack(A, B) ?
                                     scalapack::heig(A, B) : lapack::heig(A, B);
}

}

#endif // TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
