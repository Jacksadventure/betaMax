#include <re2/re2.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

static std::string trim(const std::string &s) {
    auto begin = s.begin();
    auto end = s.end();

    while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    if (begin == end) return std::string();

    do {
        --end;
    } while (end != begin && std::isspace(static_cast<unsigned char>(*end)));
    // end is at last non-space; range is [begin, end]
    return std::string(begin, end + 1);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "match_re2") << " <Category> <file_path>\n";
        return 2;
    }
    std::string category = argv[1];
    std::string file_path = argv[2];

    // Define RE2-compatible patterns, anchored for full match.
    const std::unordered_map<std::string, std::string> patterns = {
        {"Date", R"(^\d{4}-\d{2}-\d{2}$)"},                       // YYYY-MM-DD
        {"Time", R"(^\d{2}:\d{2}:\d{2}$)"},                       // HH:MM:SS
        {"URL",  R"(^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$)"},
        {"ISBN", R"(^(?:\d[- ]?){9}[\dX]$)"},                     // ISBN-10
        {"IPv4", R"(^(\d{1,3}\.){3}\d{1,3}$)"},
        {"IPv6", R"(^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$)"},
        {"FilePath", R"(^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$)"}
    };

    auto it = patterns.find(category);
    if (it == patterns.end()) {
        std::cerr << "Unknown category: " << category << "\n";
        return 2;
    }
    const std::string& pattern = it->second;

    // Read file content and trim whitespace
    std::ifstream in(file_path);
    if (!in) {
        std::cerr << "Error: File '" << file_path << "' not found.\n";
        return 1;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    std::string data = trim(oss.str());

    // Compile regex with RE2
    RE2 re(pattern);
    if (!re.ok()) {
        std::cerr << "Error: Invalid RE2 pattern for category '" << category << "'.\n";
        return 1;
    }

    // Full match required
    bool ok = RE2::FullMatch(data, re);
    return ok ? 0 : 1;
}
