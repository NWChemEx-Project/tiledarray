#ifndef TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_HEIG_H__INCLUDED
#include "TiledArray/conversions/retile.h"

namespace TiledArray {
namespace lapack {

template <typename Array>
auto heig(const Array& A) {
  using tensor_type = Array;
  using tile_type = typename tensor_type::value_type;

  auto& world = A.world();

  const auto& erange = A.elements_range();
  const auto nrows = erange.hi(0);
  const auto ncols = erange.hi(1);

  TA::TiledRange matrix_tile{{0, nrows}, {0, ncols}};
  auto A_large_tile = retile(A, matrix_tile);
  tile_type evec_buffer;
  tile_type eval_buffer;

  if (A_large_tile.is_local({0, 0})) {
    eval_buffer = tile_type(TA::Range{nrows}, 0.0);
    evec_buffer = A_large_tile.find({0, 0}).get();
    int lwork = 1 + 6 * nrows + 2 * nrows * nrows;
    int info;
    std::vector<double> work(lwork);
    dsyev("V", "U", &nrows, evec_buffer.data(), &ncols, eval_buffer.data(),
          work.data(), &lwork, &info);
    auto row_major = evec_buffer.permute(TA::Permutation{1, 0});
    evec_buffer = row_major;
  }

  auto l = [=](tile_type& tile, const auto&) {
    tile = evec_buffer;
    return tile.norm();
  };

  auto m = [=](tile_type& tile, const auto&) {
    tile = eval_buffer;
    return tile.norm();
  };

  auto pmap = A_large_tile.pmap();

  auto evecs = make_array<tensor_type>(world, matrix_tile, pmap, l);

  auto evals = make_array<tensor_type>(
      world, TA::TiledRange{matrix_tile.dim(0)}, pmap, m);

  auto tiled_evecs = retile(evecs, A.trange());
  auto tiled_evals = retile(evals, TA::TiledRange{A.trange().dim(0)});
  return std::tuple(tiled_evals, tiled_evecs);
}

}
} // namespace TiledArray

#endif
