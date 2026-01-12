#include <re2/re2.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static std::string trim(const std::string &s) {
    auto begin = s.begin();
    auto end = s.end();
    while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    if (begin == end) return std::string();
    do { --end; } while (end != begin && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(begin, end + 1);
}

static bool read_file_trim(const std::string &path, std::string &out) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) return false;
    std::ostringstream oss; oss << in.rdbuf();
    out = trim(oss.str());
    return true;
}

// Read exactly n bytes from stdin into 'out'. Returns true on success.
static bool read_n_bytes(size_t n, std::string &out) {
    out.clear();
    out.reserve(n);
    while (out.size() < n) {
        char buf[4096];
        size_t want = std::min(n - out.size(), sizeof(buf));
        std::cin.read(buf, static_cast<std::streamsize>(want));
        std::streamsize got = std::cin.gcount();
        if (got <= 0) return false;
        out.append(buf, static_cast<size_t>(got));
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "re2_server") << " <Category: Date|Time|URL|ISBN|IPv4|IPv6|FilePath>\n";
        return 2;
    }
    const std::string category = argv[1];

    const auto pattern = [&]() -> const char* {
        if (category == "Date") return R"(^\d{4}-\d{2}-\d{2}$)";
        if (category == "Time") return R"(^\d{2}:\d{2}:\d{2}$)";
        if (category == "URL")  return R"(^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$)";
        if (category == "ISBN") return R"(^(?:\d[- ]?){9}[\dX]$)";
        if (category == "IPv4") return R"(^(\d{1,3}\.){3}\d{1,3}$)";
        if (category == "IPv6") return R"(^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$)";
        if (category == "FilePath") return R"(^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$)";
        return nullptr;
    }();

    if (!pattern) {
        std::cerr << "Unknown category: " << category << "\n";
        return 2;
    }

    RE2 re(pattern);
    if (!re.ok()) {
        std::cerr << "Error: invalid RE2 pattern for category: " << category << "\n";
        return 1;
    }

    // Protocol:
    // - "FILE <path>\n"  : read file at <path>, trim, FullMatch. Respond "OK\n" or "ERR\n".
    // - "DATA <n>\n<bytes><NL>" : read exactly n bytes as data, then consume one trailing newline,
    //                            trim, FullMatch. Respond "OK\n" or "ERR\n".
    // - "QUIT\n"         : exit 0
    std::string line;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    while (std::getline(std::cin, line)) {
        if (line == "QUIT") {
            std::cout << "BYE" << std::endl;
            return 0;
        } else if (line.rfind("FILE ", 0) == 0) {
            std::string path = line.substr(5);
            std::string data;
            bool okread = read_file_trim(path, data);
            bool match = okread && RE2::FullMatch(data, re);
            std::cout << (match ? "OK" : "ERR") << std::endl;
        } else if (line.rfind("DATA ", 0) == 0) {
            const char* p = line.c_str() + 5;
            char* endp = nullptr;
            unsigned long long n = std::strtoull(p, &endp, 10);
            if (endp == p) {
                std::cout << "ERR" << std::endl;
                continue;
            }
            std::string data;
            if (!read_n_bytes(static_cast<size_t>(n), data)) {
                std::cout << "ERR" << std::endl;
                return 1;
            }
            // consume the trailing newline if present
            int c = std::cin.peek();
            if (c == '\n') {
                std::cin.get();
            }
            std::string t = trim(data);
            bool match = RE2::FullMatch(t, re);
            std::cout << (match ? "OK" : "ERR") << std::endl;
        } else {
            // Unknown command
            std::cout << "ERR" << std::endl;
        }
        std::cout.flush();
    }
    return 0;
}
