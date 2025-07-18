#include <string>

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" void oxidd$interop$std$string$assign(std::string &string,
                                                const char *data, size_t len) {
  string.assign(data, len);
}
