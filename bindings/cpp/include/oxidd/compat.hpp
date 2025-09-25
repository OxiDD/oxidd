/// @file   compat.hpp
/// @brief  Definitions for compatibility with older C++ standards

#pragma once

#include <oxidd/config.h>
#if defined(__cpp_lib_expected) && !defined(OXIDD_COMPAT_ENFORCE_TL_EXPECTED)
#include <expected>
#elif defined(OXIDD_COMPAT_ENABLE_TL_EXPECTED)
#include <tl/expected.hpp>
#else
#error "Neither std::expected nor tl::expected are available \
   (OXIDD_COMPAT_ENABLE_TL_EXPECTED was not configured)"
#endif

namespace oxidd::compat {

#if defined(__cpp_lib_expected) && !defined(OXIDD_COMPAT_ENFORCE_TL_EXPECTED)

using std::bad_expected_access;
using std::expected;
using std::unexpect;
using std::unexpect_t;
using std::unexpected;

#elif defined(OXIDD_COMPAT_ENABLE_TL_EXPECTED)

using tl::bad_expected_access;
using tl::expected;
using tl::unexpect;
using tl::unexpect_t;
using tl::unexpected;

#endif

} // namespace oxidd::compat
