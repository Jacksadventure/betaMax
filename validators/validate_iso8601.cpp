#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>

static std::string trim(const std::string &s) {
    auto begin = s.begin();
    auto end = s.end();
    while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    if (begin == end) return std::string();
    do { --end; } while (end != begin && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(begin, end + 1);
}

static bool is_word_char(char ch) {
    unsigned char uch = static_cast<unsigned char>(ch);
    return std::isalnum(uch) || ch == '_';
}

static bool parse_digits(const std::string& s, size_t& i, size_t n, int& out) {
    if (i + n > s.size()) return false;
    int v = 0;
    for (size_t k = 0; k < n; k++) {
        char ch = s[i + k];
        if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
        v = v * 10 + (ch - '0');
    }
    i += n;
    out = v;
    return true;
}

static bool parse_exact_char(const std::string& s, size_t& i, char ch) {
    if (i >= s.size() || s[i] != ch) return false;
    i++;
    return true;
}

static bool parse_fraction_no_colon_after(const std::string& s, size_t& i) {
    if (i >= s.size()) return true;
    if (s[i] != '.' && s[i] != ',') return true;
    i++;
    size_t start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    if (i == start) return false;
    if (i < s.size() && s[i] == ':') return false;
    return true;
}

static bool parse_fraction(const std::string& s, size_t& i) {
    if (i >= s.size()) return true;
    if (s[i] != '.' && s[i] != ',') return true;
    i++;
    size_t start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    return i != start;
}

static bool parse_timezone(const std::string& s, size_t& i) {
    if (i >= s.size()) return true;
    char ch = s[i];
    if (ch == 'Z' || ch == 'z') {
        i++;
        return true;
    }
    if (ch != '+' && ch != '-') return true;
    i++;
    int tz_h = 0;
    if (!parse_digits(s, i, 2, tz_h)) return false;
    if (tz_h < 0 || tz_h > 23) return false;
    bool has_colon = false;
    if (i < s.size() && s[i] == ':') {
        has_colon = true;
        i++;
    }
    // minutes are optional, even if ':' is present (matches :?([0-5]\d)?)
    if (i + 2 <= s.size() && std::isdigit(static_cast<unsigned char>(s[i])) &&
        std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
        int tz_m = 0;
        size_t j = i;
        if (!parse_digits(s, j, 2, tz_m)) return false;
        if (tz_m < 0 || tz_m > 59) return false;
        i = j;
        return true;
    }
    (void)has_colon;
    return true;
}

static bool parse_time_core(const std::string& s, size_t& i) {
    // Time core is optional at the call site; this parses it only if it starts with digits.
    if (i + 2 > s.size()) return false;
    if (!std::isdigit(static_cast<unsigned char>(s[i])) ||
        !std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
        return false;
    }

    size_t start = i;
    int hour = 0;
    if (!parse_digits(s, i, 2, hour)) return false;
    if (hour < 0 || hour > 24) return false;

    // For backref semantics:
    // - In the (hour, optional minutes) branch, separator between hour and minute (":" or "") is captured.
    // - In the (24:?00) branch, that capture group doesn't participate; backref behaves as empty.
    std::string sep_for_seconds;

    if (hour == 24) {
        // 24\:?00
        if (i < s.size() && s[i] == ':') i++;
        if (i + 2 > s.size() || s.compare(i, 2, "00") != 0) return false;
        i += 2;
        sep_for_seconds = "";
        if (!parse_fraction_no_colon_after(s, i)) return false;
    } else {
        // ((:?)[0-5]\d)?  (minutes optional, ':' optional)
        size_t j = i;
        bool minutes_present = false;
        std::string sep_minutes;

        if (j < s.size() && s[j] == ':' && j + 2 < s.size() &&
            std::isdigit(static_cast<unsigned char>(s[j + 1])) &&
            std::isdigit(static_cast<unsigned char>(s[j + 2]))) {
            j++;  // consume ':'
            int minute = 0;
            if (!parse_digits(s, j, 2, minute)) return false;
            if (minute < 0 || minute > 59) return false;
            minutes_present = true;
            sep_minutes = ":";
        } else if (j + 2 <= s.size() &&
                   std::isdigit(static_cast<unsigned char>(s[j])) &&
                   std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
            int minute = 0;
            if (!parse_digits(s, j, 2, minute)) return false;
            if (minute < 0 || minute > 59) return false;
            minutes_present = true;
            sep_minutes = "";
        }

        i = j;
        sep_for_seconds = minutes_present ? sep_minutes : "";
        if (!parse_fraction_no_colon_after(s, i)) return false;
    }

    // (\17[0-5]\d([\.,]\d+)?)?  (seconds optional)
    if (sep_for_seconds == ":") {
        if (i < s.size() && s[i] == ':') {
            i++;
            int sec = 0;
            if (!parse_digits(s, i, 2, sec)) return false;
            if (sec < 0 || sec > 59) return false;
            if (!parse_fraction(s, i)) return false;
        }
    } else {
        if (i + 2 <= s.size() &&
            std::isdigit(static_cast<unsigned char>(s[i])) &&
            std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
            int sec = 0;
            size_t j = i;
            if (!parse_digits(s, j, 2, sec)) return false;
            if (sec < 0 || sec > 59) return false;
            i = j;
            if (!parse_fraction(s, i)) return false;
        }
    }

    if (i == start) return false;
    return true;
}

static bool parse_date_and_time(const std::string& s, size_t& i) {
    // Parse optional date part after year:
    // (-?) (month/day | week | ordinal) (optional time part)
    char date_sep = '\0';
    if (i < s.size() && s[i] == '-') {
        date_sep = '-';
        i++;
    }

    auto parse_time_part = [&](size_t& k) -> bool {
        if (k >= s.size()) return true;
        char ch = s[k];
        if (ch != 'T' && !std::isspace(static_cast<unsigned char>(ch))) return true;
        k++;  // consume exactly one 'T' or whitespace char

        // Inner group is optional: can end right after the separator.
        if (k >= s.size()) return true;

        // Optional time core
        size_t j = k;
        if (parse_time_core(s, j)) {
            k = j;
        }

        // Optional timezone
        if (!parse_timezone(s, k)) return false;
        return true;
    };

    auto try_parse_full = [&](auto&& parse_date_fn) -> bool {
        size_t j = i;
        if (!parse_date_fn(j)) return false;
        if (!parse_time_part(j)) return false;
        if (j != s.size()) return false;
        i = j;
        return true;
    };

    // 1) Month/day: (0[1-9]|1[0-2])(\3(day))?
    auto parse_month_day = [&](size_t& j) -> bool {
        int month = 0;
        if (!parse_digits(s, j, 2, month)) return false;
        if (month < 1 || month > 12) return false;

        auto try_with_day = [&](bool require_day) -> bool {
            size_t k = j;
            if (require_day) {
                if (date_sep == '-') {
                    if (!parse_exact_char(s, k, '-')) return false;
                }
                int day = 0;
                if (!parse_digits(s, k, 2, day)) return false;
                if (day < 1 || day > 31) return false;
            }
            j = k;
            return true;
        };

        // Prefer taking the optional day when it's syntactically possible, but backtrack if needed.
        size_t k = j;
        bool day_possible = false;
        if (date_sep == '-') {
            day_possible = (k < s.size() && s[k] == '-' && k + 2 < s.size() &&
                            std::isdigit(static_cast<unsigned char>(s[k + 1])) &&
                            std::isdigit(static_cast<unsigned char>(s[k + 2])));
        } else {
            day_possible = (k + 2 <= s.size() &&
                            std::isdigit(static_cast<unsigned char>(s[k])) &&
                            std::isdigit(static_cast<unsigned char>(s[k + 1])));
        }

        if (day_possible) {
            if (try_with_day(true)) return true;
        }
        return try_with_day(false);
    };

    // 2) Week date: W(00-52)(-?[1-7])?
    auto parse_week = [&](size_t& j) -> bool {
        if (!parse_exact_char(s, j, 'W')) return false;
        int week = 0;
        if (!parse_digits(s, j, 2, week)) return false;
        if (week < 0 || week > 52) return false;
        if (j >= s.size()) return true;
        if (s[j] == '-') {
            if (j + 1 >= s.size()) return true;
            char d = s[j + 1];
            if (d >= '1' && d <= '7') {
                j += 2;
                return true;
            }
            return false;
        }
        char d = s[j];
        if (d >= '1' && d <= '7') {
            j += 1;
        }
        return true;
    };

    // 3) Ordinal date: 001-366
    auto parse_ordinal = [&](size_t& j) -> bool {
        int day_of_year = 0;
        if (!parse_digits(s, j, 3, day_of_year)) return false;
        return day_of_year >= 1 && day_of_year <= 366;
    };

    // Try in the same order as the regex alternation: month/day, week, ordinal.
    if (try_parse_full(parse_month_day)) return true;
    if (try_parse_full(parse_week)) return true;
    if (try_parse_full(parse_ordinal)) return true;
    return false;
}

static bool is_iso8601_like(const std::string& s) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;

    int year = 0;
    if (!parse_digits(s, i, 4, year)) return false;

    // (?!\d{2}\b) after the year.
    if (i + 2 <= s.size() &&
        std::isdigit(static_cast<unsigned char>(s[i])) &&
        std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
        if (i + 2 == s.size()) return false;  // boundary at end
        if (!is_word_char(s[i + 2])) return false;  // boundary before non-word
    }

    // Entire remaining group is optional; if present, it must parse and consume fully.
    if (i == s.size()) return true;

    size_t j = i;
    if (!parse_date_and_time(s, j)) return false;
    return j == s.size();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "validate_iso8601") << " <file_path>\n";
        return 2;
    }
    const char* file_path = argv[1];

    std::ifstream in(file_path);
    if (!in) {
        std::cerr << "Error: File '" << file_path << "' not found.\n";
        return 1;
    }
    std::ostringstream oss; oss << in.rdbuf();
    std::string data = trim(oss.str());

    return is_iso8601_like(data) ? 0 : 1;
}
