#ifndef TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
#define TILEDARRAY_MATH_LINEAR_ALGEBRA_H__INCLUDED
#include "TiledArray/math/scalapack.h"
#include "TiledArray/math/lapack/heig.h"
#include "TiledArray/config.h"

namespace TiledArray {

template<typename Array>
auto heig(const Array& A) {
  // TODO: Check if tensor is distributed, if not don't use SCALAPACK

#if TILEDARRAY_HAS_SCALAPACK
  return scalapack::heig(A);
#else
  return lapack::heig(A);
}

template<typename Array>
auto heig(const Array& A, const Array& B) {
  // TODO: Check if tensor is distributed, if not don't use SCALAPACK

#if TILEDARRAY_HAS_SCALAPACK
  return scalapack::heig(A, B);
#else
  return lapack::heig(A, B);
}

}
