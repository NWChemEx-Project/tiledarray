#ifndef TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#include "TiledArray/conversions/eigen.h"
#include <madness/tensor/linalg_wrapper.h>
namespace TiledArray::lapack {

template <typename Array>
auto heig(const Array& A) {
  using tensor_type = Array;
  using numeric_type = typename tensor_type::numeric_type;
  using tile_type = typename tensor_type::value_type;

  auto& world = A.world();

  auto A_eigen = array_to_eigen(A);
  const int nrows = A_eigen.rows();
  const int ncols = A_eigen.cols();

  int lwork = -1;
  int info;
  std::vector<numeric_type> work(1);
  std::vector<numeric_type> evals(nrows);
  madness::hereig('V', 'U', nrows, A_eigen.data(), ncols, evals.data(), work.data());
  const auto& matrix_trange = A.trange();
  TiledRange vector_trange{matrix_trange.dim(0)};

  auto evecs =
    column_major_buffer_to_array<tensor_type>(world, matrix_trange,
                                                A_eigen.data(), nrows, ncols);
  return std::tuple(evals, evecs);
}

template <typename Array>
auto heig(const Array& A, const Array& B) {
  using tensor_type = Array;
  using numeric_type = typename tensor_type::numeric_type;
  using tile_type = typename tensor_type::value_type;

  auto& world = A.world();

  auto A_eigen = array_to_eigen(A);
  auto B_eigen = array_to_eigen(B);
  const int nrows = A_eigen.rows();
  const int ncols = A_eigen.cols();

  int lwork = -1;
  int liwork = -1;
  int info;
  std::vector<numeric_type> evals(nrows);
  std::vector<numeric_type> work(1);
  std::vector<int> iwork(1);

  madness::hereig_gen('V', 'U', nrows, A_eigen.data(), ncols, evals.data(), work.data());

  const auto& matrix_trange = A.trange();
  TiledRange vector_trange{matrix_trange.dim(0)};

  auto evecs =
      column_major_buffer_to_array<tensor_type>(world, matrix_trange,
                                                A_eigen.data(), nrows, ncols);

  return std::tuple(evals, evecs);
}

} // namespace TiledArray::lapack

#endif
