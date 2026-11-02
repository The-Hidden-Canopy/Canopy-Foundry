#pragma once

// Keep Windows' wide command-line paths intact when existing parsers consume
// UTF-8 char arguments. Non-Windows entry points continue using native argv.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ida_native {
class Utf8Arguments {
public:
    Utf8Arguments(const Utf8Arguments&) = delete;
    Utf8Arguments& operator=(const Utf8Arguments&) = delete;
    Utf8Arguments(Utf8Arguments&&) = delete;
    Utf8Arguments& operator=(Utf8Arguments&&) = delete;
    Utf8Arguments(int argc, wchar_t** argv) {
        if (argc < 0 || (argc && !argv)) throw std::runtime_error("invalid Windows arguments");
        values_.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            if (!argv[i]) throw std::runtime_error("invalid Windows argument");
            const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
            if (size <= 0) throw std::runtime_error("Windows argument is not valid Unicode");
            std::string value(static_cast<std::size_t>(size), '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr) != size)
                throw std::runtime_error("Windows argument conversion failed");
            value.pop_back();
            values_.push_back(std::move(value));
        }
        pointers_.reserve(values_.size() + 1);
        for (auto& value : values_) pointers_.push_back(value.data());
        pointers_.push_back(nullptr);
    }
    char** data() noexcept { return pointers_.data(); }
private:
    std::vector<std::string> values_;
    std::vector<char*> pointers_;
};
}
#endif
