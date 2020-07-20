#ifndef TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
#include "TiledArray/conversions/eigen.h"

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
  int info;
  if constexpr (std::is_same_v<numeric_type, double>) {
    dpotrf("L", &nrows, A_eigen.data(), &ncols, &info);
  } else {
    TA_EXCEPTION("Your numeric type is not hooked up at the moment");
  }

  // I think I need to zero out the upper triangle, but I'm not 100% sure...
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
  using numeric_type = typename tensor_type::numeric_type;

  auto L_eigen = detail::cholesky_(A);
  int nrows = L_eigen.rows();
  int ncols = L_eigen.cols();
  int info;
  if constexpr(std::is_same_v<numeric_type, double>){
    dtrtri("L", "N", &nrows, L_eigen.data(), &ncols, &info);
  } else {
    TA_EXCEPTION("Your numeric type is not hooked up at the moment");
  }
  for(auto i = 0; i < nrows; ++i)
    for(auto j = i + 1; j < ncols; ++j) L_eigen(i,j) = 0.0;

  return column_major_buffer_to_array<tensor_type>(
      A.world(), A.trange(), L_eigen.data(), nrows, ncols);
}

template<typename Array>
auto cholesky_solve(const Array& A, const Array& B) {
  TA_EXCEPTION("LAPACK version is not implemented");
}


} // namespace TiledArray::lapack

#endif // TILEDARRAY_MATH_LAPACK_CHOL_H__INCLUDED
