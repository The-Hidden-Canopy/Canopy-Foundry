#include "ida_native/utf8_arguments.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <stdexcept>

// Synthetic argument conversion only; no owner, CUDA or training process.
int main() {
#ifdef _WIN32
    wchar_t program[] = L"foundry.exe";
    wchar_t unicode_path[] = L"C:\\fixture\\\x03A9\\request.json";
    wchar_t* argv[] = {program, unicode_path};
    ida_native::Utf8Arguments arguments(2, argv);
    assert(arguments.data()[2] == nullptr);
    assert(std::filesystem::u8path(arguments.data()[1]) == std::filesystem::path(unicode_path));
    wchar_t invalid[] = {static_cast<wchar_t>(0xD800), L'\0'};
    wchar_t* invalid_argv[] = {invalid};
    bool rejected = false;
    try { ida_native::Utf8Arguments bad(1, invalid_argv); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
#endif
    return 0;
}
