#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

#include "TiledArray/math/lapack.h"

struct LAPACKFixture {
  size_t              mat_dim;
  std::vector<double> ref_matrix;  // XXX: Just double is fine?

  std::vector<double> htoeplitz_vector;
  std::vector<double> exact_evals;

  inline
  double matrix_element_generator( int64_t i, int64_t j ) {
    
    #if 0 
    // Generates a Hankel matrix: absurd condition number
    return i+j;
    #else
    // Generates a Circulant matrix: good condition number
    return htoeplitz_vector[std::abs(i-j)];
    #endif
  }

  inline
  double make_ta_reference(TA::Tensor<double>& t,
                                  TA::Range const& range) {
    t = TA::Tensor<double>(range, 0.0);
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m) {
      for (auto n = lo[1]; n < up[1]; ++n) {
        t(m, n) = matrix_element_generator(m,n);
      }
    }

    return t.norm();
  };

  inline void construct_lapack( size_t N, std::vector<double>& A ) {
    if(A.size() != (N*N)) A.resize(N*N);
    for (size_t j = 0; j < N; ++j)
    for (size_t i = 0; i < N; ++i)
      A[i + j*N] = matrix_element_generator(i,j);

  }

  LAPACKFixture(int64_t N)
      : mat_dim(N), ref_matrix(N*N, 0),
        htoeplitz_vector( N ), exact_evals( N )	{

    // Generate an hermitian Circulant vector
    std::fill( htoeplitz_vector.begin(), htoeplitz_vector.begin(), 0 );
    htoeplitz_vector[0] = 100;
    std::default_random_engine gen(0);
    std::uniform_real_distribution<> dist(0., 1.);
    for( int64_t i = 1; i <= (N/2); ++i ) {
      double val = dist(gen);
      htoeplitz_vector[i]   = val;
      htoeplitz_vector[N-i] = val;
    }

    // Compute exact eigenvalues
    const double ff = 2. * M_PI / N;
    for( int64_t j = 0; j < N; ++j ) {
      double val = htoeplitz_vector[0];;
      for( int64_t k = 1; k < N; ++k )
        val += htoeplitz_vector[N-k] * std::cos( ff * j * k );
      exact_evals[j] = val;
    }

    std::sort( exact_evals.begin(), exact_evals.end() );

    // Fill reference matrix
    construct_lapack( N, ref_matrix );
  }

  LAPACKFixture() : LAPACKFixture(10) {}
};

TA::TiledRange gen_trange(size_t N, const std::vector<size_t>& TA_NBs) {
  assert(TA_NBs.size() > 0);

  std::default_random_engine gen(0);
  std::uniform_int_distribution<> dist(0, TA_NBs.size() - 1);
  auto rand_indx = [&]() { return dist(gen); };
  auto rand_nb = [&]() { return TA_NBs[rand_indx()]; };

  std::vector<size_t> t_boundaries = {0};
  auto TA_NB = rand_nb();
  while (t_boundaries.back() + TA_NB < N) {
    t_boundaries.emplace_back(t_boundaries.back() + TA_NB);
    TA_NB = rand_nb();
  }
  t_boundaries.emplace_back(N);

  std::vector<TA::TiledRange1> ranges(
      2, TA::TiledRange1(t_boundaries.begin(), t_boundaries.end()));

  return TA::TiledRange(ranges.begin(), ranges.end());
};

BOOST_FIXTURE_TEST_SUITE(lapack_suite, LAPACKFixture)



BOOST_AUTO_TEST_CASE( lapack_heig ) {

  if( GlobalFixture::world->size() != 1 ) return; 

  GlobalFixture::world->gop.fence();
  auto N = mat_dim;
  auto trange = gen_trange(N, {2ul});

  const auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [evals, evecs] = lapack::heig( ref_ta );

  BOOST_CHECK( evecs.trange() == ref_ta.trange() );

  // TODO: Check validity of eigenvectors, not crutial for the time being

  // Check eigenvalue correctness
  double tol = N*N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(evals[i] - exact_evals[i]), tol );

  GlobalFixture::world->gop.fence();
}


#if 0
BOOST_AUTO_TEST_CASE( sca_heig_generalized ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TSpArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto dense_iden = TA::make_array<TA::TSpArray<double> >(
      *GlobalFixture::world, trange,
      [](TA::Tensor<double>& t, TA::Range const& range) -> double {
        t = TA::Tensor<double>(range, 0.0);
        auto lo = range.lobound_data();
        auto up = range.upbound_data();
        for (auto m = lo[0]; m < up[0]; ++m) 
        for (auto n = lo[1]; n < up[1]; ++n) 
	if( m == n )
          t(m, n) = 1.;
	
	return t.norm();
      });

  auto [evals, evecs] = heig( ref_ta, dense_iden );
  //auto evals = heig( ref_ta );

  BOOST_CHECK( evecs.trange() == ref_ta.trange() );

  // TODO: Check validity of eigenvectors, not crutial for the time being

  // Check eigenvalue correctness
  double tol = N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(evals[i] - exact_evals[i]), tol );

  GlobalFixture::world->gop.fence();
}



BOOST_AUTO_TEST_CASE( sca_chol ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto L = cholesky( ref_ta );

  BOOST_CHECK( L.trange() == ref_ta.trange() );

  ref_ta("i,j") -= L("i,k") * L("j,k").conj();

  double diff_norm = ref_ta("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( diff_norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}



BOOST_AUTO_TEST_CASE( sca_chol_linv ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

     
  auto Linv = cholesky_linv( ref_ta );

  BOOST_CHECK( Linv.trange() == ref_ta.trange() );

  TA::TArray<double> tmp( *GlobalFixture::world, trange );
  tmp("i,j")    = Linv("i,k") * ref_ta("k,j");
  ref_ta("i,j") = tmp ("i,k") * Linv  ("j,k");

  TA::foreach_inplace( ref_ta, []( TA::Tensor<double>& tile ) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
    for (auto n = lo[1]; n < up[1]; ++n) 
    if( m == n ) {
      tile(m,n) -= 1.;
    }
  });

  double norm = ref_ta("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}




BOOST_AUTO_TEST_CASE( sca_chol_linv_retl ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

     
  auto [L, Linv] = cholesky_linv<decltype(ref_ta),true>( ref_ta );

  BOOST_CHECK( Linv.trange() == ref_ta.trange() );
  BOOST_CHECK( L.trange()    == ref_ta.trange() );

  TA::TArray<double> tmp( *GlobalFixture::world, trange );
  tmp("i,j") = Linv("i,k") * L("k,j");

  TA::foreach_inplace( tmp, []( TA::Tensor<double>& tile ) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
    for (auto n = lo[1]; n < up[1]; ++n) 
    if( m == n ) {
      tile(m,n) -= 1.;
    }
  });

  double norm = tmp("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}



BOOST_AUTO_TEST_CASE( sca_chol_solve ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

     
  auto iden = cholesky_solve( ref_ta, ref_ta );

  BOOST_CHECK( iden.trange() == ref_ta.trange() );

  TA::foreach_inplace( iden, []( TA::Tensor<double>& tile ) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
    for (auto n = lo[1]; n < up[1]; ++n) 
    if( m == n ) {
      tile(m,n) -= 1.;
    }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}





BOOST_AUTO_TEST_CASE( sca_chol_lsolve ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

     
  // Should produce X = L**H
  auto [L, X] = cholesky_lsolve( lapackpp::TransposeFlag::NoTranspose,
    ref_ta, ref_ta );

  BOOST_CHECK( X.trange() == ref_ta.trange() );
  BOOST_CHECK( L.trange() == ref_ta.trange() );

  X("i,j") -= L("j,i");

  double norm = X("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE( sca_lu_solve ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

     
  auto iden = lu_solve( ref_ta, ref_ta );

  BOOST_CHECK( iden.trange() == ref_ta.trange() );

  TA::foreach_inplace( iden, []( TA::Tensor<double>& tile ) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
    for (auto n = lo[1]; n < up[1]; ++n) 
    if( m == n ) {
      tile(m,n) -= 1.;
    }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}


BOOST_AUTO_TEST_CASE( sca_lu_inv ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  TA::TArray<double> iden( *GlobalFixture::world, trange );
     
  auto Ainv = lu_inv( ref_ta );
  iden("i,j") = Ainv("i,k") * ref_ta("k,j");


  BOOST_CHECK( iden.trange() == ref_ta.trange() );

  TA::foreach_inplace( iden, []( TA::Tensor<double>& tile ) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
    for (auto n = lo[1]; n < up[1]; ++n) 
    if( m == n ) {
      tile(m,n) -= 1.;
    }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL( norm, N*N*std::numeric_limits<double>::epsilon() );

  GlobalFixture::world->gop.fence();
}


#if 1
BOOST_AUTO_TEST_CASE( sca_svd_values_only ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto S = svd<SVDValuesOnly>( ref_ta, trange, trange );

  std::vector exact_singular_values = exact_evals;
  std::sort( exact_singular_values.begin(), exact_singular_values.end(),
    std::greater<double>() );

  // Check singular value correctness
  double tol = N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(S[i] - exact_singular_values[i]), tol );
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE( sca_svd_leftvectors ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S,U] = svd<SVDLeftVectors>( ref_ta, trange, trange );

  std::vector exact_singular_values = exact_evals;
  std::sort( exact_singular_values.begin(), exact_singular_values.end(),
    std::greater<double>() );

  // Check singular value correctness
  double tol = N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(S[i] - exact_singular_values[i]), tol );
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE( sca_svd_rightvectors ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S,VT] = svd<SVDRightVectors>( ref_ta, trange, trange );

  std::vector exact_singular_values = exact_evals;
  std::sort( exact_singular_values.begin(), exact_singular_values.end(),
    std::greater<double>() );

  // Check singular value correctness
  double tol = N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(S[i] - exact_singular_values[i]), tol );
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE( sca_svd_allvectors ) {

  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S,U,VT] = svd<SVDAllVectors>( ref_ta, trange, trange );

  std::vector exact_singular_values = exact_evals;
  std::sort( exact_singular_values.begin(), exact_singular_values.end(),
    std::greater<double>() );

  // Check singular value correctness
  double tol = N*N*std::numeric_limits<double>::epsilon();
  for( int64_t i = 0; i < N; ++i )
    BOOST_CHECK_SMALL( std::abs(S[i] - exact_singular_values[i]), tol );
  GlobalFixture::world->gop.fence();
}
#endif


#endif

BOOST_AUTO_TEST_SUITE_END()
