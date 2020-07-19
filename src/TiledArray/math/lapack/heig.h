#ifndef TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#include "TiledArray/conversions/eigen.h"

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

  int lwork = 1 + 6 * nrows + 2 * nrows * nrows;
  int info;
  std::vector<numeric_type> work(lwork);
  std::vector<numeric_type> eval_buffer(nrows);

  if constexpr(std::is_same_v<numeric_type, double>) {
    dsyev("V", "U", &nrows, A_eigen.data(), &ncols, eval_buffer.data(),
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

  auto l = [=](tile_type& tile, const auto& range) {
    tile_type buffer(range);
    for(auto i : range) buffer[i] = eval_buffer[i[0]];
    tile = buffer;
    return tile.norm();
  };
  auto evals = make_array<tensor_type>(world, vector_trange, l);

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

  int lwork = 1 + 6 * nrows + 2 * nrows * nrows;
  int liwork = 3 + 5 * nrows;
  int info;
  std::vector<numeric_type> work(lwork);
  std::vector<numeric_type> eval_buffer(nrows);
  std::vector<int> iwork(liwork);

  if constexpr(std::is_same_v<numeric_type, double>) {
    int one = 1;

    dsygvd(&one, "V", "U", &nrows, A_eigen.data(), &ncols, B_eigen.data(),
           &ncols, eval_buffer.data(), work.data(), &lwork, iwork.data(),
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

  auto l = [=](tile_type& tile, const auto& range) {
    tile_type buffer(range);
    for(auto i : range) buffer[i] = eval_buffer[i[0]];
    tile = buffer;
    return tile.norm();
  };
  auto evals = make_array<tensor_type>(world, vector_trange, l);

  return std::tuple(evals, evecs);
}

} // namespace TiledArray::lapack

#endif
