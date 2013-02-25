/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_CYCLIC_PMAP_H__INCLUDED
#define TILEDARRAY_CYCLIC_PMAP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/pmap.h>
#include <cmath>
#include <TiledArray/madness.h>

namespace TiledArray {
  namespace detail {

    /// Map processes using cyclic decomposition

    /// This map does a cyclic decomposition of an m-by-n matrix. It distributes
    /// processors into an x-by-y matrix such that the ratios of m/n and x/y are
    /// approximately equal.
    class CyclicPmap : public Pmap<std::size_t> {
    public:
      typedef Pmap<std::size_t>::key_type key_type;
      typedef Pmap<std::size_t>::const_iterator const_iterator;

      CyclicPmap(madness::World& world, std::size_t m, std::size_t n) :
          rank_(world.rank()),
          procs_(world.size()),
          m_(m),
          n_(n),
          x_(),
          y_(),
          local_()
      {
        TA_ASSERT(m > 0ul);
        TA_ASSERT(n > 0ul);

        // Get a rough estimate of the process dimensions
        // The ratios of x_ / y_  and m_ / n_ should be approximately equal.
        // Constraints: 1 <= x_ <= procs_ && 1 <= y_
        // The process map should be no bigger than m * n
        y_ = procs_ / std::max(std::min<std::size_t>(
            std::sqrt(procs_ * m_ / n_), procs_), 1ul);
        x_ = procs_ / y_;

        // Maximum size is m and n
        x_ = std::min(x_, m_);
        y_ = std::min(y_, n_);

        TA_ASSERT(x_ * y_ <= procs_);
      }

      CyclicPmap(madness::World& world, std::size_t m, std::size_t n, std::size_t x, std::size_t y) :
          rank_(world.rank()),
          procs_(world.size()),
          m_(m),
          n_(n),
          x_(std::min<std::size_t>(x,m)),
          y_(std::min<std::size_t>(y,n)),
          local_()
      {
        TA_ASSERT(x_ * y_ <= procs_);
      }

    private:

      CyclicPmap(const CyclicPmap& other) :
          rank_(other.rank_),
          procs_(other.procs_),
          m_(other.m_),
          n_(other.n_),
          x_(other.x_),
          y_(other.y_),
          local_()
      { }

    public:

      virtual ~CyclicPmap() { }

      /// Set the hashing seed
      virtual void set_seed(madness::hashT = 0) {
        // Construct the local process map.
        local_.reserve((m_ / x_) * (n_ / y_));
        const key_type size = x_ * y_;
        const std::size_t end = m_ * n_;
        for(std::size_t r = 0ul; r < size; ++r) {
            if(r == rank_) {
              const std::size_t m_step = n_ * x_;
              const std::size_t row_end_offset = n_ - (r % y_);
              for(std::size_t m = (r / y_) * n_ + (r % y_); m < end; m += m_step) {
                const std::size_t row_end = m + row_end_offset;
                for(std::size_t i = m; i < row_end; i += y_)
                  local_.push_back(i);
              }
            }
        }
      }


      /// Create a copy of this pmap

      /// \return A shared pointer to the new object
      virtual std::shared_ptr<Pmap<key_type> > clone() const {
        return std::shared_ptr<Pmap<key_type> >(new CyclicPmap(*this));
      }

      /// Key owner

      /// \param key The element key to be mapped
      /// \return The \c ProcessID of the process that owns \c key .
      virtual ProcessID owner(const key_type& key) const {
        TA_ASSERT(key < (m_ * n_));
//        // Get matrix coordinate
//        const std::size_t m = key / n_;
//        const std::size_t n = key % n_;
//        // Get processor coordinate
//        const std::size_t x = m % x_;
//        const std::size_t y = n % y_;
//        // Get processor ordinal
//        const std::size_t o = x * y_ + y;
        const std::size_t o = ((key / n_) % x_) * y_ + ((key % n_) % y_);

        TA_ASSERT(o < procs_);

        return o;
      }

      /// Local size accessor

      /// \return The number of local elements
      virtual std::size_t local_size() const { return local_.size(); }

      /// Local elements

      /// \return \c true when there are no local elements, otherwise \c false .
      virtual bool empty() const { return local_.empty(); }

      /// Begin local element iterator

      /// \return An iterator that points to the beginning of the local element set
      virtual const_iterator begin() const { return local_.begin(); }


      /// End local element iterator

      /// \return An iterator that points to the beginning of the local element set
      virtual const_iterator end() const { return local_.end(); }

      /// Local element vector accessor

      /// \return A const reference to a vector of local elements
      virtual const std::vector<key_type>& local() const { return local_; }

    private:

      std::size_t procs_; ///< Number of processes in the world
      std::size_t rank_; ///< This process's rank
      std::size_t m_; ///< Number of rows to be mapped
      std::size_t n_; ///< Number of columns to be mapped
      std::size_t x_; ///< Number of process rows
      std::size_t y_; ///< Number of process columns
      madness::hashT seed_; ///< Hashing seed for process randomization
      std::vector<key_type> local_; ///< A vector of local elements in map.
    }; // class CyclicPmap

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_CYCLIC_PMAP_H__INCLUDED
