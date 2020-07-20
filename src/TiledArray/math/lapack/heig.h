#ifndef TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#include "TiledArray/conversions/eigen.h"
#include <madness/tensor/clapack.h>
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

  if constexpr(std::is_same_v<numeric_type, double>) {
    dsyev_("V", "U", &nrows, A_eigen.data(), &ncols, evals.data(),
          work.data(), &lwork, &info);
    lwork = work[0];
    work = std::vector<numeric_type>(lwork);
    dsyev_("V", "U", &nrows, A_eigen.data(), &ncols, evals.data(),
          work.data(), &lwork, &info);
  }
  else {
    TA_EXCEPTION("Your numeric type is not hooked up at the moment");
  }

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

  if constexpr(std::is_same_v<numeric_type, double>) {
    int one = 1;
    // First call gets optimial sizes
    dsygvd_(&one, "V", "U", &nrows, A_eigen.data(), &ncols, B_eigen.data(),
           &ncols, evals.data(), work.data(), &lwork, iwork.data(),
           &liwork, &info);
    lwork = work[0];
    liwork = iwork[0];
    work = std::vector<numeric_type>(lwork);
    iwork = std::vector<int>(liwork);
    // This call does the real work
    dsygvd_(&one, "V", "U", &nrows, A_eigen.data(), &ncols, B_eigen.data(),
           &ncols, evals.data(), work.data(), &lwork, iwork.data(),
           &liwork, &info);

  }
  else {
    TA_EXCEPTION("Your numeric type is not hooked up at the moment");
  }

  const auto& matrix_trange = A.trange();
  TiledRange vector_trange{matrix_trange.dim(0)};

  auto evecs =
      column_major_buffer_to_array<tensor_type>(world, matrix_trange,
                                                A_eigen.data(), nrows, ncols);

  return std::tuple(evals, evecs);
}

} // namespace TiledArray::lapack

#endif
