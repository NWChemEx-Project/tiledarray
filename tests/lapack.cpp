#include "tiledarray.h"
#include "TiledArray/math/lapack/chol.h"
#include "TiledArray/math/lapack/heig.h"
#include "unit_test_config.h"

using namespace TiledArray;

BOOST_AUTO_TEST_SUITE(heig_orthogonal)

BOOST_AUTO_TEST_CASE(two_by_two) {
  auto& world = get_default_world();
  TiledRange trange{{0, 1, 2}, {0, 1, 2}};
  TSpArrayD a(world, trange, {{1.23, 4.56}, {4.56, 7.89}});
  auto [eval, evecs] = lapack::heig(a);

  TSpArrayD eval_corr(world, TiledRange{{0, 1, 2}}, {-1.08645907, 10.20645907});
  TSpArrayD evec_corr(world, trange, {{-0.89155766,  0.4529072},
                                      {0.4529072 ,  0.89155766}});

  std::cout << eval << std::endl;
  std::cout << evecs << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(heig_general)

BOOST_AUTO_TEST_CASE(two_by_two) {
  auto& world = get_default_world();
  TiledRange trange{{0, 1, 2}, {0, 1, 2}};
  TSpArrayD a(world, trange, {{1.23, 4.56}, {4.56, 7.89}});
  TSpArrayD b(world, trange, {{1.00, 0.50}, {0.50, 1.00}});
  auto [eval, evecs] = lapack::heig(a, b);

  TSpArrayD eval_corr(world, TiledRange{{0, 1, 2}}, {-1.86171399,  7.94171399});
  TSpArrayD evec_corr(world, trange, {{-1.15165094, 0.0838657},
                                      {0.6484553, 0.95542611}});
  std::cout << eval << std::endl;
  std::cout << evecs << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(chol)

BOOST_AUTO_TEST_CASE(two_by_two) {
  auto& world = get_default_world();
  TiledRange trange{{0, 1, 2}, {0, 1, 2}};
  TSpArrayD a(world, trange, {{1.00, 1.00}, {1.00, 2.00}});
  auto L = lapack::cholesky(a);
  std::cout << L << std::endl;

  TSpArrayD L_corr(world, trange, {{1.0, 0.0}, {1.0, 1.0}});
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(chol_inv)

BOOST_AUTO_TEST_CASE(two_by_two) {
  auto& world = get_default_world();
  TiledRange trange{{0, 1, 2}, {0, 1, 2}};
  TSpArrayD a(world, trange, {{1.00, 1.00}, {1.00, 2.00}});
  auto linv = lapack::cholesky_linv(a);
  std::cout << linv << std::endl;

  TSpArrayD ainv_corr(world, trange, {{1.0, 0.0}, {-1.0, 1.0}});
}

BOOST_AUTO_TEST_SUITE_END()
