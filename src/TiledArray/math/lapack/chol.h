#ifndef TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
#include "TiledArray/conversions/eigen.h"
#include <madness/tensor/clapack.h>
#include <madness/tensor/linalg_wrappers.h>

namespace TiledArray::lapack {
namespace detail {

/// \brief Peforms Cholesky, but does not convert back to a DistArray
///
/// Cholesky decomposition is the first step of `cholesky_linv` and
/// `cholesky_solve`, but if we try to write them in terms of `cholesky` we'll
/// end up creating a DistArray with L, only to have to take it apart again.
/// This function is the guts of the `cholesky` function without the conversion
/// to a DistArray.
///
/// @tparam Array
/// @param A
/// @return
template <typename Array>
auto cholesky_(const Array& A) {
  using tensor_type = Array;
  using numeric_type = typename tensor_type::numeric_type;

  auto A_eigen = array_to_eigen(A);
  int nrows = A_eigen.rows();
  int ncols = A_eigen.cols();
  if( nrows != ncols ) TA_EXCEPTION("Cholesky Requires Square");

  madness::cholesky('L', nrows, A_eigen.data(), nrows);

  // I think I need to zero out the upper triangle, but I'm not 100% sure...
  for (auto i = 0; i < nrows; ++i)
    for (auto j = i + 1; j < ncols; ++j) A_eigen(i, j) = 0.0;
  return A_eigen;
}

template <typename Array>
auto cholesky_linv_( const Array& A ) {
  using tensor_type = Array;
  using numeric_type = typename tensor_type::numeric_type;

  auto A_eigen = array_to_eigen(A);
  int nrows = A_eigen.rows();
  int ncols = A_eigen.cols();
  if( nrows != ncols ) TA_EXCEPTION("Cholesky Requires Square");

  madness::cholesky('L', nrows, A_eigen.data(), nrows);
  madness::trtri('L', 'N', nrows, A_eigen.data(), nrows);

  for (auto i = 0; i < nrows; ++i)
    for (auto j = i + 1; j < ncols; ++j) A_eigen(i, j) = 0.0;
  return A_eigen;
}

} // namespace detail


template <typename Array>
auto cholesky(const Array& A) {
  using tensor_type = Array;
  auto L_eigen = detail::cholesky_(A);
  const auto nrows = A.elements_range().upbound(0);
  return column_major_buffer_to_array<tensor_type>(
      A.world(), A.trange(), L_eigen.data(), nrows, nrows);
}

template<typename Array>
auto cholesky_linv(const Array& A) {
  using tensor_type = Array;
  auto Linv_eigen = detail::cholesky_linv_(A);
  const auto nrows = A.elements_range().upbound(0);
  return column_major_buffer_to_array<tensor_type>(
      A.world(), A.trange(), Linv_eigen.data(), nrows, nrows);
}

template<typename Array>
auto cholesky_solve(const Array& A, const Array& B) {
  TA_EXCEPTION("LAPACK version is not implemented");
}


} // namespace TiledArray::lapack

#endif // TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
