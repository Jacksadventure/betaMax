#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace fs = std::filesystem;

static std::string trim_cr(std::string s) {
  if (!s.empty() && s.back() == '\r') s.pop_back();
  return s;
}

static std::vector<std::string> read_lines(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open: " + path.string());
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) out.push_back(trim_cr(line));
  return out;
}

static std::string read_file_all_strip_one_trailing_newline(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open: " + path.string());
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (!content.empty() && content.back() == '\n') content.pop_back();
  return content;
}

static std::string getenv_str(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static bool getenv_bool(const char* name) {
  auto v = getenv_str(name);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return (v == "1" || v == "true" || v == "yes");
}

static std::string preview_for_log(std::string_view s, size_t max_len = 200) {
  std::string out;
  out.reserve(std::min(max_len, s.size()) + 32);
  for (size_t i = 0; i < s.size() && i < max_len; i++) {
    unsigned char c = (unsigned char)s[i];
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else if (std::isprint(c)) {
      out.push_back((char)c);
    } else {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\x%02X", (unsigned)c);
      out += buf;
    }
  }
  if (s.size() > max_len) out += "...(truncated)";
  return out;
}

// Minimal shlex-like splitter (supports single/double quotes and backslash escaping).
static std::vector<std::string> split_cmdline(std::string_view s) {
  std::vector<std::string> out;
  std::string cur;
  enum class Q { None, Single, Double };
  Q q = Q::None;
  bool esc = false;
  auto flush = [&]() {
    if (!cur.empty()) out.push_back(cur);
    cur.clear();
  };
  for (char ch : s) {
    if (esc) {
      cur.push_back(ch);
      esc = false;
      continue;
    }
    if (q != Q::Single && ch == '\\') {
      esc = true;
      continue;
    }
    if (q == Q::None) {
      if (std::isspace((unsigned char)ch)) {
        flush();
        continue;
      }
      if (ch == '\'') {
        q = Q::Single;
        continue;
      }
      if (ch == '"') {
        q = Q::Double;
        continue;
      }
      cur.push_back(ch);
      continue;
    }
    if (q == Q::Single) {
      if (ch == '\'') {
        q = Q::None;
      } else {
        cur.push_back(ch);
      }
      continue;
    }
    if (q == Q::Double) {
      if (ch == '"') {
        q = Q::None;
      } else {
        cur.push_back(ch);
      }
      continue;
    }
  }
  flush();
  return out;
}

struct Options {
  fs::path repo_root = ".";
  std::optional<fs::path> positives;
  std::optional<fs::path> negatives;
  std::optional<std::string> broken;
  std::optional<fs::path> broken_file;
  std::optional<fs::path> output_file;
  std::optional<fs::path> dfa_cache;
  bool init_cache = false;
  std::string category;
  std::optional<std::string> oracle_validator;
  std::string learner = "rpni";
  int xover_pairs = -1;   // -1 => read env / default
  int xover_checks = -1;  // -1 => read env / default
  int mutations = 0;
  int mutations_edits = 1;
  bool mutations_random = true;
  bool mutations_deterministic = false;
  std::optional<uint64_t> mutations_seed;
  int max_attempts = 500;
  int attempt_candidates = 1;
  int max_cost = 3;
  int max_candidates = 50;
  int oracle_timeout_ms = 3000;
  bool eq_disable_sampling = false;
  int eq_max_length = 10;
  int eq_samples_per_length = 20;
  int eq_max_oracle = 0;
  int eq_max_rounds = 0;
  std::optional<uint64_t> seed;
  bool verbose = false;
};

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage (repair):\n"
      << "  " << argv0 << " --positives <file> --category <Cat> (--broken <s> | --broken-file <f>) [options]\n"
      << "Usage (precompute DFA cache):\n"
      << "  " << argv0 << " --positives <file> --category <Cat> --dfa-cache <path> --init-cache [options]\n"
      << "Options:\n"
      << "  --positives <path>\n"
      << "  --negatives <path>\n"
      << "  --broken <string>\n"
      << "  --broken-file <path>\n"
      << "  --output-file <path>\n"
      << "  --dfa-cache <path>              (load/save DFA cache)\n"
      << "  --init-cache                    (learn DFA and write --dfa-cache, then exit)\n"
      << "  --category <Date|ISO8601|Time|URL|ISBN|IPv4|IPv6|FilePath>\n"
      << "  --oracle-validator <cmd>\n"
      << "  --learner <rpni|rpni_xover>     (default: rpni)\n"
      << "  --xover-pairs <int>             (default env LSTAR_RPNI_XOVER_PAIRS or 50; 0 disables)\n"
      << "  --xover-checks <int>            (default env LSTAR_RPNI_XOVER_CHECKS or 10; 0 disables)\n"
      << "  --mutations <int>               (default: 0)\n"
      << "  --mutations-edits <int>         (default: 1)\n"
      << "  --mutations-random              (default)\n"
      << "  --mutations-deterministic       (enumerate single-edit neighbors, capped by --mutations)\n"
      << "  --mutations-seed <uint64>       (optional)\n"
      << "  --max-attempts <int>            (default: 500)\n"
      << "  --attempt-candidates <int>      (default: 1)\n"
      << "  --repo-root <path>              (default: .)\n"
      << "  --max-cost <int>                (default: 3; -1 disables the fixed cap)\n"
      << "  --max-candidates <int>          (default: 50)\n"
      << "  --oracle-timeout-ms <int>       (default: 3000)\n"
      << "  --eq-disable-sampling           (disable oracle-based negative sampling)\n"
      << "  --eq-max-length <int>           (default: 10)\n"
      << "  --eq-samples-per-length <int>   (default: 20)\n"
      << "  --eq-max-oracle <int>           (default: 0)\n"
      << "  --eq-max-rounds <int>           (default: 0)\n"
      << "  --seed <uint64>                 (random seed)\n"
      << "  --verbose                       (more debug output)\n";
}

static std::optional<Options> parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return std::nullopt;
    } else if (a == "--positives") {
      opt.positives = fs::path(need("--positives"));
    } else if (a == "--negatives") {
      opt.negatives = fs::path(need("--negatives"));
    } else if (a == "--broken") {
      opt.broken = need("--broken");
    } else if (a == "--broken-file") {
      opt.broken_file = fs::path(need("--broken-file"));
    } else if (a == "--output-file") {
      opt.output_file = fs::path(need("--output-file"));
    } else if (a == "--dfa-cache") {
      opt.dfa_cache = fs::path(need("--dfa-cache"));
    } else if (a == "--init-cache") {
      opt.init_cache = true;
    } else if (a == "--category") {
      opt.category = need("--category");
    } else if (a == "--oracle-validator") {
      opt.oracle_validator = need("--oracle-validator");
    } else if (a == "--learner") {
      opt.learner = need("--learner");
    } else if (a == "--xover-pairs") {
      opt.xover_pairs = std::stoi(need("--xover-pairs"));
    } else if (a == "--xover-checks") {
      opt.xover_checks = std::stoi(need("--xover-checks"));
    } else if (a == "--mutations") {
      opt.mutations = std::stoi(need("--mutations"));
    } else if (a == "--mutations-edits") {
      opt.mutations_edits = std::stoi(need("--mutations-edits"));
    } else if (a == "--mutations-random") {
      opt.mutations_random = true;
      opt.mutations_deterministic = false;
    } else if (a == "--mutations-deterministic") {
      opt.mutations_deterministic = true;
      opt.mutations_random = false;
    } else if (a == "--mutations-seed") {
      opt.mutations_seed = (uint64_t)std::stoull(need("--mutations-seed"));
    } else if (a == "--max-attempts") {
      opt.max_attempts = std::stoi(need("--max-attempts"));
    } else if (a == "--attempt-candidates") {
      opt.attempt_candidates = std::stoi(need("--attempt-candidates"));
    } else if (a == "--repo-root") {
      opt.repo_root = fs::path(need("--repo-root"));
    } else if (a == "--max-cost") {
      opt.max_cost = std::stoi(need("--max-cost"));
    } else if (a == "--max-candidates") {
      opt.max_candidates = std::stoi(need("--max-candidates"));
    } else if (a == "--oracle-timeout-ms") {
      opt.oracle_timeout_ms = std::stoi(need("--oracle-timeout-ms"));
    } else if (a == "--eq-disable-sampling") {
      opt.eq_disable_sampling = true;
    } else if (a == "--eq-max-length") {
      opt.eq_max_length = std::stoi(need("--eq-max-length"));
    } else if (a == "--eq-samples-per-length") {
      opt.eq_samples_per_length = std::stoi(need("--eq-samples-per-length"));
    } else if (a == "--eq-max-oracle") {
      opt.eq_max_oracle = std::stoi(need("--eq-max-oracle"));
    } else if (a == "--eq-max-rounds") {
      opt.eq_max_rounds = std::stoi(need("--eq-max-rounds"));
    } else if (a == "--seed") {
      opt.seed = (uint64_t)std::stoull(need("--seed"));
    } else if (a == "--verbose" || a == "-v") {
      opt.verbose = true;
    } else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }
  if (getenv_bool("BETAMAX_VERBOSE")) opt.verbose = true;
  if (!opt.positives) throw std::runtime_error("--positives is required");
  if (opt.category.empty()) throw std::runtime_error("--category is required");
  if (opt.init_cache) {
    if (!opt.dfa_cache) throw std::runtime_error("--dfa-cache is required with --init-cache");
    if (opt.broken || opt.broken_file) throw std::runtime_error("--init-cache does not take --broken/--broken-file");
  } else {
    if ((bool)opt.broken == (bool)opt.broken_file) {
      throw std::runtime_error("provide exactly one of --broken or --broken-file");
    }
  }
  // Allow max_cost = -1 as "unbounded".
  if (opt.max_cost < -1) opt.max_cost = -1;
  if (opt.max_candidates < 1) opt.max_candidates = 1;
  if (opt.oracle_timeout_ms < 1) opt.oracle_timeout_ms = 1;
  if (opt.mutations < 0) opt.mutations = 0;
  if (opt.mutations_edits < 1) opt.mutations_edits = 1;
  if (opt.max_attempts < 1) opt.max_attempts = 1;
  if (opt.attempt_candidates < 1) opt.attempt_candidates = 1;
  if (opt.eq_max_length < 0) opt.eq_max_length = 0;
  if (opt.eq_samples_per_length < 0) opt.eq_samples_per_length = 0;
  if (opt.eq_max_oracle < 0) opt.eq_max_oracle = 0;
  if (opt.eq_max_rounds < 0) opt.eq_max_rounds = 0;
  return opt;
}

struct DFA {
  int start = 0;
  std::vector<std::unordered_map<char, int>> trans;
  std::vector<unsigned char> accept;

  bool accepts(std::string_view s) const {
    int st = start;
    for (char ch : s) {
      auto it = trans[st].find(ch);
      if (it == trans[st].end()) return false;
      st = it->second;
    }
    return accept[st] != 0;
  }
};

static bool write_dfa_cache(const fs::path& path, const DFA& dfa) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "BMXDFA1\n";
  out << dfa.start << "\n";
  out << dfa.trans.size() << "\n";
  out << dfa.accept.size() << "\n";
  size_t acc_count = 0;
  for (size_t i = 0; i < dfa.accept.size(); i++) {
    if (dfa.accept[i]) acc_count++;
  }
  out << acc_count;
  for (size_t i = 0; i < dfa.accept.size(); i++) {
    if (dfa.accept[i]) out << " " << i;
  }
  out << "\n";
  size_t tcount = 0;
  for (size_t s = 0; s < dfa.trans.size(); s++) tcount += dfa.trans[s].size();
  out << tcount << "\n";
  for (size_t s = 0; s < dfa.trans.size(); s++) {
    for (const auto& kv : dfa.trans[s]) {
      int byte = (int)(unsigned char)kv.first;
      out << s << " " << byte << " " << kv.second << "\n";
    }
  }
  out.flush();
  return (bool)out;
}

static std::optional<DFA> read_dfa_cache(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::string magic;
  if (!std::getline(in, magic)) return std::nullopt;
  if (magic != "BMXDFA1") return std::nullopt;
  int start = 0;
  size_t nstates = 0;
  size_t naccept = 0;
  if (!(in >> start)) return std::nullopt;
  if (!(in >> nstates)) return std::nullopt;
  if (!(in >> naccept)) return std::nullopt;
  if (nstates == 0 || naccept != nstates) return std::nullopt;

  size_t acc_count = 0;
  if (!(in >> acc_count)) return std::nullopt;
  std::vector<unsigned char> accept(nstates, 0);
  for (size_t i = 0; i < acc_count; i++) {
    size_t idx = 0;
    if (!(in >> idx)) return std::nullopt;
    if (idx >= nstates) return std::nullopt;
    accept[idx] = 1;
  }

  size_t tcount = 0;
  if (!(in >> tcount)) return std::nullopt;
  std::vector<std::unordered_map<char, int>> trans(nstates);
  for (size_t i = 0; i < tcount; i++) {
    size_t from = 0;
    int byte = 0;
    int to = 0;
    if (!(in >> from >> byte >> to)) return std::nullopt;
    if (from >= nstates) return std::nullopt;
    if (to < 0 || (size_t)to >= nstates) return std::nullopt;
    if (byte < 0 || byte > 255) return std::nullopt;
    char ch = (char)(unsigned char)byte;
    trans[from][ch] = to;
  }

  DFA dfa;
  dfa.start = start;
  if (dfa.start < 0 || (size_t)dfa.start >= nstates) return std::nullopt;
  dfa.trans = std::move(trans);
  dfa.accept = std::move(accept);
  return dfa;
}

struct PTA {
  struct Node {
    int id = 0;
    bool accept = false;
    std::unordered_map<char, int> next;
    int parent = -1;
    char via = 0;
  };

  std::vector<Node> nodes;
  std::unordered_set<char> alphabet;

  PTA() { nodes.push_back(Node{0, false, {}, -1, 0}); }

  int add_path(std::string_view w, bool is_positive) {
    int s = 0;
    for (char ch : w) {
      alphabet.insert(ch);
      auto it = nodes[(size_t)s].next.find(ch);
      if (it == nodes[(size_t)s].next.end()) {
        int nid = (int)nodes.size();
        nodes.push_back(Node{nid, false, {}, s, ch});
        nodes[(size_t)s].next.emplace(ch, nid);
        s = nid;
      } else {
        s = it->second;
      }
    }
    if (is_positive) nodes[(size_t)s].accept = true;
    return s;
  }
};

class RPNI {
 public:
  RPNI(std::vector<std::string> positives, std::vector<std::string> negatives)
      : positives_(std::move(positives)), negatives_(std::move(negatives)) {
    for (const auto& w : positives_) pta_.add_path(w, true);
    for (const auto& w : negatives_) {
      for (char ch : w) pta_.alphabet.insert(ch);
    }
  }

  virtual ~RPNI() = default;

  DFA learn() {
    const int n = (int)pta_.nodes.size();
    std::vector<int> rep(n);
    for (int i = 0; i < n; i++) rep[i] = i;

    std::set<int> red;
    std::set<int> blue;
    auto add_blue_of = [&](int r) {
      for (const auto& kv : pta_.nodes[(size_t)r].next) {
        int v = kv.second;
        if (!red.count(v)) blue.insert(v);
      }
    };

    red.insert(0);
    add_blue_of(0);

    while (!blue.empty()) {
      int qb = *blue.begin();
      blue.erase(blue.begin());

      bool merged = false;
      for (int qr : red) {
        auto rep_try = try_merge(rep, qr, qb);
        if (rep_try) {
          rep = std::move(*rep_try);
          merged = true;
          break;
        }
      }
      (void)merged;
      red.insert(qb);
      add_blue_of(qb);
    }

    DFA dfa = materialize(rep);
    if (!consistent_with_negatives(dfa)) {
      // Should not happen; fallback to PTA DFA.
      std::vector<int> id(n);
      for (int i = 0; i < n; i++) id[i] = i;
      return materialize(id);
    }
    return dfa;
  }

 protected:
  virtual bool can_merge(const std::vector<int>& rep, int qr, int qb) {
    (void)rep;
    (void)qr;
    (void)qb;
    return true;
  }

  int find(const std::vector<int>& rep, int v) const {
    int r = v;
    while (rep[(size_t)r] != r) r = rep[(size_t)r];
    return r;
  }

  DFA materialize(const std::vector<int>& rep_in) const {
    const int n = (int)pta_.nodes.size();
    std::vector<int> canon = rep_in;
    for (int v = 0; v < n; v++) {
      while (canon[(size_t)v] != canon[(size_t)canon[(size_t)v]]) {
        canon[(size_t)v] = canon[(size_t)canon[(size_t)v]];
      }
    }

    std::unordered_map<int, int> idmap;
    idmap.reserve((size_t)n);
    int idc = 0;
    for (int v = 0; v < n; v++) {
      int r = canon[(size_t)v];
      if (!idmap.count(r)) idmap.emplace(r, idc++);
    }

    DFA dfa;
    dfa.start = idmap.at(canon[0]);
    dfa.trans.assign((size_t)idc, {});
    dfa.accept.assign((size_t)idc, 0);

    for (int v = 0; v < n; v++) {
      int r = idmap.at(canon[(size_t)v]);
      if (pta_.nodes[(size_t)v].accept) dfa.accept[(size_t)r] = 1;
    }

    for (int v = 0; v < n; v++) {
      int r = idmap.at(canon[(size_t)v]);
      for (const auto& kv : pta_.nodes[(size_t)v].next) {
        char a = kv.first;
        int u = kv.second;
        int ru = idmap.at(canon[(size_t)u]);
        dfa.trans[(size_t)r][a] = ru;
      }
    }
    return dfa;
  }

  bool consistent_with_negatives(const DFA& dfa) const {
    if (negatives_.empty()) return true;
    for (const auto& w : negatives_) {
      if (dfa.accepts(w)) return false;
    }
    return true;
  }

  std::optional<std::vector<int>> try_merge(const std::vector<int>& rep_in, int qr, int qb) {
    if (!can_merge(rep_in, qr, qb)) return std::nullopt;

    std::vector<int> rep = rep_in;
    rep[(size_t)qb] = qr;

    std::deque<std::pair<int, int>> q;
    q.emplace_back(qr, qb);

    while (!q.empty()) {
      auto [x, y] = q.front();
      q.pop_front();
      for (const auto& kv : pta_.nodes[(size_t)y].next) {
        char a = kv.first;
        int ny = kv.second;
        auto it_x = pta_.nodes[(size_t)x].next.find(a);
        if (it_x == pta_.nodes[(size_t)x].next.end()) continue;
        int nx = it_x->second;

        int rx = find(rep, nx);
        int ry = find(rep, ny);
        if (rx != ry) {
          rep[(size_t)ry] = rx;
          q.emplace_back(rx, ry);
        }
      }
    }

    if (negatives_.empty()) return rep;

    DFA dfa = materialize(rep);
    if (!consistent_with_negatives(dfa)) return std::nullopt;
    return rep;
  }

  PTA pta_;
  std::vector<std::string> positives_;
  std::vector<std::string> negatives_;
};

class XoverRPNI final : public RPNI {
 public:
  using MembershipOracle = std::function<bool(std::string_view)>;

  XoverRPNI(
      std::vector<std::string> positives,
      std::vector<std::string> negatives,
      MembershipOracle is_member,
      int max_pairs,
      int max_checks)
      : RPNI(std::move(positives), std::move(negatives)),
        is_member_(std::move(is_member)),
        max_pairs_(std::max(0, max_pairs)),
        max_checks_(std::max(0, max_checks)) {
    compute_node_prefixes();
    index_positive_suffixes();
    index_negative_strings();
  }

 protected:
  bool can_merge(const std::vector<int>& rep, int qr, int qb) override {
    if (!is_member_ || max_pairs_ <= 0 || max_checks_ <= 0) return true;

    int root_r = find(rep, qr);
    int root_b = find(rep, qb);
    if (root_r == root_b) return true;

    int budget = max_checks_;

    // 1) Cross-over from positives: p(qr) · s(qb) using one suffix.
    if (budget > 0) {
      std::string_view prefix = (qr >= 0 && (size_t)qr < node_prefix_.size()) ? std::string_view(node_prefix_[(size_t)qr]) : std::string_view();
      const auto it = pos_suffixes_.find(qb);
      std::string_view suffix = (it != pos_suffixes_.end() && !it->second.empty()) ? std::string_view(it->second[0]) : std::string_view();

      std::string cand;
      cand.reserve(prefix.size() + suffix.size());
      cand.append(prefix.data(), prefix.size());
      cand.append(suffix.data(), suffix.size());
      if (!cand.empty()) {
        auto v = oracle_accepts(cand);
        budget--;
        if (v.has_value() && v.value() == false) return false;
      }
    }

    // 2) Negatives that pass through either node must stay negative.
    if (budget > 0) {
      std::unordered_set<std::string> seen;
      auto check_list = [&](int node) -> bool {
        auto it = negatives_by_node_.find(node);
        if (it == negatives_by_node_.end()) return true;
        for (const auto& cand : it->second) {
          if (budget <= 0) return true;
          if (!seen.insert(cand).second) continue;
          auto v = oracle_accepts(cand);
          budget--;
          if (v.has_value() && v.value() == true) return false;
        }
        return true;
      };
      if (!check_list(qr)) return false;
      if (!check_list(qb)) return false;
    }

    return true;
  }

 private:
  std::optional<bool> oracle_accepts(std::string_view word) {
    std::string key(word);
    auto it = oracle_cache_.find(key);
    if (it != oracle_cache_.end()) return it->second;
    bool verdict = false;
    try {
      verdict = bool(is_member_(word));
    } catch (...) {
      return std::nullopt;
    }
    oracle_cache_.emplace(std::move(key), verdict);
    return verdict;
  }

  void compute_node_prefixes() {
    node_prefix_.assign(pta_.nodes.size(), "");
    for (size_t node_id = 1; node_id < pta_.nodes.size(); node_id++) {
      int p = pta_.nodes[node_id].parent;
      char via = pta_.nodes[node_id].via;
      if (p >= 0 && (size_t)p < node_prefix_.size()) {
        node_prefix_[node_id] = node_prefix_[(size_t)p] + std::string(1, via);
      } else {
        node_prefix_[node_id] = std::string(1, via);
      }
    }
  }

  void index_positive_suffixes() {
    for (const auto& w : positives_) {
      int node = 0;
      pos_suffixes_[node].push_back(w);
      size_t consumed = 0;
      for (char ch : w) {
        auto it = pta_.nodes[(size_t)node].next.find(ch);
        if (it == pta_.nodes[(size_t)node].next.end()) break;
        node = it->second;
        consumed++;
        pos_suffixes_[node].push_back(w.substr(consumed));
      }
    }
  }

  void index_negative_strings() {
    for (const auto& w : negatives_) {
      int node = 0;
      negatives_by_node_[node].push_back(w);
      for (char ch : w) {
        auto it = pta_.nodes[(size_t)node].next.find(ch);
        if (it == pta_.nodes[(size_t)node].next.end()) break;
        node = it->second;
        negatives_by_node_[node].push_back(w);
      }
    }
  }

  MembershipOracle is_member_;
  int max_pairs_ = 50;
  int max_checks_ = 10;

  std::vector<std::string> node_prefix_;
  std::unordered_map<int, std::vector<std::string>> pos_suffixes_;
  std::unordered_map<int, std::vector<std::string>> negatives_by_node_;
  std::unordered_map<std::string, bool> oracle_cache_;
};

static std::string category_to_base(std::string_view cat) {
  if (cat == "Date") return "date";
  if (cat == "Time") return "time";
  if (cat == "URL") return "url";
  if (cat == "ISBN") return "isbn";
  if (cat == "IPv4") return "ipv4";
  if (cat == "IPv6") return "ipv6";
  if (cat == "FilePath") return "pathfile";
  if (cat == "ISO8601") return "iso8601";
  std::string out(cat);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return out;
}

static std::vector<char> derive_alphabet_from_positives(const std::vector<std::string>& positives) {
  std::unordered_set<char> s;
  for (const auto& w : positives) {
    for (char ch : w) s.insert(ch);
  }
  std::vector<char> out(s.begin(), s.end());
  std::sort(out.begin(), out.end());
  return out;
}

static std::string apply_random_edit(
    std::string_view base,
    const std::vector<char>& alphabet,
    std::mt19937_64& rng) {
  // Operations: 0=delete, 1=insert, 2=substitute
  std::uniform_int_distribution<int> op_dist(0, 2);
  int op = op_dist(rng);
  if (alphabet.empty()) op = 0;

  std::string s(base);

  if (op == 0) {  // delete
    if (s.empty()) return s;
    std::uniform_int_distribution<size_t> pos_dist(0, s.size() - 1);
    size_t pos = pos_dist(rng);
    s.erase(pos, 1);
    return s;
  }

  if (op == 1) {  // insert
    std::uniform_int_distribution<size_t> pos_dist(0, s.size());
    std::uniform_int_distribution<size_t> ch_dist(0, alphabet.size() - 1);
    size_t pos = pos_dist(rng);
    char ch = alphabet[ch_dist(rng)];
    s.insert(s.begin() + (std::ptrdiff_t)pos, ch);
    return s;
  }

  // substitute
  if (s.empty()) return s;
  std::uniform_int_distribution<size_t> pos_dist(0, s.size() - 1);
  std::uniform_int_distribution<size_t> ch_dist(0, alphabet.size() - 1);
  size_t pos = pos_dist(rng);
  char ch = alphabet[ch_dist(rng)];
  s[pos] = ch;
  return s;
}

static std::vector<std::string> generate_mutations_random(
    const std::vector<std::string>& positives,
    int n,
    int edits,
    const std::vector<char>& alphabet,
    std::mt19937_64& rng) {
  std::vector<std::string> out;
  out.reserve((size_t)std::max(0, n));
  if (n <= 0 || positives.empty()) return out;

  std::uniform_int_distribution<size_t> pick_pos(0, positives.size() - 1);
  std::unordered_set<std::string> seen;
  seen.reserve((size_t)n * 2 + 8);

  int tries = 0;
  int max_tries = std::max(200, n * 30);
  while ((int)out.size() < n && tries < max_tries) {
    tries++;
    const std::string& base = positives[pick_pos(rng)];
    std::string cur = base;
    for (int e = 0; e < edits; e++) {
      cur = apply_random_edit(cur, alphabet, rng);
    }
    if (cur == base) continue;
    if (!seen.insert(cur).second) continue;
    out.push_back(std::move(cur));
  }
  return out;
}

static std::vector<std::string> generate_mutations_deterministic(
    const std::vector<std::string>& positives,
    int n,
    const std::vector<char>& alphabet) {
  std::vector<std::string> out;
  out.reserve((size_t)std::max(0, n));
  if (n <= 0) return out;

  std::unordered_set<std::string> seen;
  seen.reserve((size_t)n * 2 + 8);

  std::vector<std::string> pos_sorted = positives;
  std::sort(pos_sorted.begin(), pos_sorted.end());

  for (const auto& w : pos_sorted) {
    if ((int)out.size() >= n) break;
    // delete
    for (size_t i = 0; i < w.size() && (int)out.size() < n; i++) {
      std::string s = w;
      s.erase(i, 1);
      if (s == w) continue;
      if (seen.insert(s).second) out.push_back(std::move(s));
    }
    if ((int)out.size() >= n) break;
    // substitute
    for (size_t i = 0; i < w.size() && (int)out.size() < n; i++) {
      for (char ch : alphabet) {
        if ((int)out.size() >= n) break;
        if (w[i] == ch) continue;
        std::string s = w;
        s[i] = ch;
        if (seen.insert(s).second) out.push_back(std::move(s));
      }
    }
    if ((int)out.size() >= n) break;
    // insert
    for (size_t i = 0; i <= w.size() && (int)out.size() < n; i++) {
      for (char ch : alphabet) {
        if ((int)out.size() >= n) break;
        std::string s = w;
        s.insert(s.begin() + (std::ptrdiff_t)i, ch);
        if (seen.insert(s).second) out.push_back(std::move(s));
      }
    }
  }
  return out;
}

struct Oracle {
  fs::path repo_root;
  std::string category;
  std::optional<std::vector<std::string>> override_cmd;  // argv without temp file
  int timeout_ms = 3000;
  bool debug = false;
  bool verbose = false;

  struct Stats {
    uint64_t total = 0;
    uint64_t correct = 0;
    uint64_t incorrect = 0;
    uint64_t incomplete = 0;
    double seconds_total = 0.0;
  };
  mutable Stats stats;

  std::vector<std::string> default_cmd_argv(const fs::path& temp_path) const {
    std::string base = category_to_base(category);
    // Prefer pure-C++ validators first (no Python dependency), then regex wrappers.
    fs::path c1 = repo_root / "validators" / ("validate_" + base);
    fs::path c2 = repo_root / "validators" / "regex" / ("validate_" + base);
    if (fs::exists(c1)) return {c1.string(), temp_path.string()};
    if (fs::exists(c2)) return {c2.string(), temp_path.string()};
    return {"python3", (repo_root / "match.py").string(), category, temp_path.string()};
  }

  bool validate_text(std::string_view text) const {
    const auto t0 = std::chrono::steady_clock::now();
    auto record = [&](bool ok, bool incomplete, const char* why) -> bool {
      stats.total++;
      if (incomplete) {
        stats.incomplete++;
      } else if (ok) {
        stats.correct++;
      } else {
        stats.incorrect++;
      }
      const auto t1 = std::chrono::steady_clock::now();
      stats.seconds_total += std::chrono::duration<double>(t1 - t0).count();
      if (verbose) {
        std::cerr << "[DEBUG] Membership verdict: "
                  << (incomplete ? "INCOMPLETE" : (ok ? "ACCEPT" : "REJECT")) << " for '"
                  << preview_for_log(text) << "'";
        if (why && *why) std::cerr << " (" << why << ")";
        std::cerr << "\n";
      }
      return ok;
    };

    // temp file
    fs::path dir = fs::temp_directory_path();
    std::string tmpl = (dir / "betamax_cpp_XXXXXX.txt").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');

#if defined(_WIN32)
    // Simple fallback: write to temp name and run via system (no timeout).
    char tmpname[L_tmpnam];
    std::tmpnam(tmpname);
    fs::path temp_path = dir / tmpname;
    {
      std::ofstream out(temp_path, std::ios::binary);
      out.write(text.data(), (std::streamsize)text.size());
    }
    std::vector<std::string> argv = override_cmd ? *override_cmd : default_cmd_argv(temp_path);
    if (override_cmd) argv.push_back(temp_path.string());
    std::string cmdline;
    for (const auto& s : argv) {
      if (!cmdline.empty()) cmdline.push_back(' ');
      cmdline += s;
    }
    int rc = std::system(cmdline.c_str());
    std::error_code ec;
    fs::remove(temp_path, ec);
    return record(rc == 0, false, "system");
#else
    int fd = ::mkstemps(buf.data(), 4);  // ".txt"
    if (fd < 0) return record(false, true, "mkstemps");
    fs::path temp_path(buf.data());
    // Write content.
    ssize_t w = ::write(fd, text.data(), text.size());
    (void)w;
    ::close(fd);

    std::vector<std::string> argv = override_cmd ? *override_cmd : default_cmd_argv(temp_path);
    if (override_cmd) argv.push_back(temp_path.string());

    if (debug) {
      std::cerr << "[DEBUG] Oracle in: len=" << text.size() << "\n";
      std::cerr << "[DEBUG] Oracle cmd:";
      for (const auto& s : argv) std::cerr << " " << s;
      std::cerr << "\n";
    }

    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (auto& s : argv) cargv.push_back(const_cast<char*>(s.c_str()));
    cargv.push_back(nullptr);

    pid_t pid = 0;
    int spawn_rc = ::posix_spawnp(&pid, cargv[0], nullptr, nullptr, cargv.data(), environ);
    if (spawn_rc != 0) {
      if (debug) std::cerr << "[WARN] posix_spawnp failed: " << std::strerror(spawn_rc) << "\n";
      std::error_code ec;
      fs::remove(temp_path, ec);
      return record(false, true, "posix_spawnp");
    }

    auto start = std::chrono::steady_clock::now();
    int status = 0;
    while (true) {
      pid_t r = ::waitpid(pid, &status, WNOHANG);
      if (r == pid) break;
      if (r == 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
          if (debug) std::cerr << "[WARN] Oracle timed out after " << timeout_ms << "ms; killing.\n";
          ::kill(pid, SIGKILL);
          (void)::waitpid(pid, &status, 0);
          std::error_code ec;
          fs::remove(temp_path, ec);
          return record(false, true, "timeout");
        }
        ::usleep(2000);
        continue;
      }
      // waitpid error
      if (debug) std::cerr << "[WARN] waitpid error: " << std::strerror(errno) << "\n";
      break;
    }

    std::error_code ec;
    fs::remove(temp_path, ec);

    if (WIFEXITED(status)) {
      int rc = WEXITSTATUS(status);
      if (debug) std::cerr << "[DEBUG] Oracle rc: " << rc << "\n";
      return record(rc == 0, false, "exit");
    }
    if (debug) std::cerr << "[DEBUG] Oracle terminated abnormally\n";
    return record(false, true, "abnormal");
#endif
  }
};

struct RepairCandidate {
  int cost = 0;
  std::string text;
};

struct PathStep {
  int prev = -1;
  char emitted = 0;  // 0 means no char emitted (delete)
};

static std::string materialize_path(const std::vector<PathStep>& steps, int path_id) {
  std::string out;
  int id = path_id;
  while (id >= 0) {
    char c = steps[id].emitted;
    if (c != 0) out.push_back(c);
    id = steps[id].prev;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

static std::vector<RepairCandidate> generate_candidates_to_dfa(
    const DFA& dfa,
    std::string_view broken,
    int max_cost,
    int max_candidates) {
  // Cost-ordered search on product graph (state, pos). Each node stores a persistent output path.
  struct Node {
    int cost;
    int state;
    int pos;
    int path_id;
  };
  struct Cmp {
    bool operator()(const Node& a, const Node& b) const { return a.cost > b.cost; }
  };

  std::priority_queue<Node, std::vector<Node>, Cmp> pq;
  std::vector<PathStep> steps;
  steps.push_back(PathStep{-1, 0});  // path_id=0 is empty

  auto push_emit = [&](int prev_path, char emitted) -> int {
    steps.push_back(PathStep{prev_path, emitted});
    return (int)steps.size() - 1;
  };

  // Allow multiple distinct paths per (state,pos,cost) to avoid collapsing away
  // oracle-valid solutions that share the same product node.
  int max_paths_per_key = max_candidates * 5;
  if (max_paths_per_key < 30) max_paths_per_key = 30;
  if (max_paths_per_key > 2000) max_paths_per_key = 2000;
  const size_t key_w = broken.size() + 1;
  const size_t key_cost = (size_t)max_cost + 1;
  auto key = [&](int st, int pos, int cost) -> size_t {
    return ((size_t)st * key_w + (size_t)pos) * key_cost + (size_t)cost;
  };
  std::vector<uint16_t> seen(dfa.trans.size() * key_w * key_cost, 0);
  const size_t push_budget = (size_t)std::max(1000, max_candidates * 20000);
  size_t pushed = 0;

  pq.push(Node{0, dfa.start, 0, 0});
  seen[key(dfa.start, 0, 0)] = 1;

  std::vector<RepairCandidate> out;

  while (!pq.empty()) {
    Node cur = pq.top();
    pq.pop();
    if (cur.cost > max_cost) break;

    if (cur.pos == (int)broken.size() && dfa.accept[cur.state]) {
      out.push_back(RepairCandidate{cur.cost, materialize_path(steps, cur.path_id)});
      if ((int)out.size() >= max_candidates) break;
      // Continue: there may be other solutions with same cost.
    }

    auto try_push = [&](int ncost, int nstate, int npos, int npath) {
      if (ncost > max_cost) return;
      if (pushed >= push_budget) return;
      size_t k = key(nstate, npos, ncost);
      if ((int)seen[k] >= max_paths_per_key) return;
      seen[k] = (uint16_t)(seen[k] + 1);
      pq.push(Node{ncost, nstate, npos, npath});
      pushed++;
    };

    // Delete: consume input without emitting.
    if (cur.pos < (int)broken.size()) {
      int ncost = cur.cost + 1;
      int npos = cur.pos + 1;
      try_push(ncost, cur.state, npos, cur.path_id);
    }

    // Insert or match/substitute: follow transitions and emit char.
    const auto& m = dfa.trans[cur.state];
    for (const auto& kv : m) {
      char c = kv.first;
      int ns = kv.second;
      // Insert
      {
        int ncost = cur.cost + 1;
        try_push(ncost, ns, cur.pos, push_emit(cur.path_id, c));
      }
      // Match/Substitute
      if (cur.pos < (int)broken.size()) {
        int add = (broken[(size_t)cur.pos] == c) ? 0 : 1;
        int ncost = cur.cost + add;
        int npos = cur.pos + 1;
        try_push(ncost, ns, npos, push_emit(cur.path_id, c));
      }
    }
  }

  // De-dup (can happen with multiple paths yielding same output).
  std::unordered_set<std::string> dedup_seen;
  std::vector<RepairCandidate> dedup;
  dedup.reserve(out.size());
  for (auto& c : out) {
    if (dedup_seen.insert(c.text).second) dedup.push_back(std::move(c));
  }
  return dedup;
}

static std::vector<RepairCandidate> generate_candidates_to_dfa_unbounded(
    const DFA& dfa,
    std::string_view broken,
    int max_candidates) {
  // Cost-ordered search on product graph (state, pos) without a fixed max_cost cap.
  struct Node {
    int cost;
    int state;
    int pos;
    int path_id;
  };
  struct Cmp {
    bool operator()(const Node& a, const Node& b) const { return a.cost > b.cost; }
  };

  std::priority_queue<Node, std::vector<Node>, Cmp> pq;
  std::vector<PathStep> steps;
  steps.push_back(PathStep{-1, 0});  // path_id=0 is empty

  auto push_emit = [&](int prev_path, char emitted) -> int {
    steps.push_back(PathStep{prev_path, emitted});
    return (int)steps.size() - 1;
  };

  const size_t key_w = broken.size() + 1;
  const size_t push_budget = (size_t)std::max(1000, max_candidates * 20000);
  size_t pushed = 0;
  size_t popped = 0;

  // Allow multiple distinct paths per (state,pos,cost) to reduce over-collapsing.
  int max_paths_per_key = max_candidates * 5;
  if (max_paths_per_key < 30) max_paths_per_key = 30;
  if (max_paths_per_key > 2000) max_paths_per_key = 2000;

  // Seen-count per (state,pos,cost) packed key. Keeps memory bounded by push_budget.
  std::unordered_map<uint64_t, uint16_t> seen;
  seen.reserve(push_budget / 4 + 64);

  auto pack_key = [&](int st, int pos, int cost) -> uint64_t {
    // cost: 22 bits, st: 21 bits, pos: 21 bits
    return (uint64_t)(cost & ((1 << 22) - 1)) << 42 |
           (uint64_t)(st & ((1 << 21) - 1)) << 21 |
           (uint64_t)(pos & ((1 << 21) - 1));
  };

  auto try_push = [&](int ncost, int nstate, int npos, int npath) {
    if (ncost < 0) return;
    if (ncost >= (1 << 22)) return;  // avoid pack overflow; far beyond any practical edit distance here
    if (pushed >= push_budget) return;
    if ((size_t)nstate >= dfa.trans.size()) return;
    if (npos < 0 || (size_t)npos >= key_w) return;

    uint64_t k = pack_key(nstate, npos, ncost);
    auto it = seen.find(k);
    if (it != seen.end() && (int)it->second >= max_paths_per_key) return;
    if (it == seen.end()) {
      seen.emplace(k, 1);
    } else {
      it->second = (uint16_t)(it->second + 1);
    }
    pq.push(Node{ncost, nstate, npos, npath});
    pushed++;
  };

  pq.push(Node{0, dfa.start, 0, 0});
  seen.emplace(pack_key(dfa.start, 0, 0), 1);

  std::vector<RepairCandidate> out;
  out.reserve((size_t)std::max(1, max_candidates));
  const int want = std::max(1, max_candidates);
  const int collect = std::min(want * 3, want + 200);  // oversample before de-dup
  int last_solution_cost = -1;

  while (!pq.empty()) {
    Node cur = pq.top();
    pq.pop();
    popped++;
    if (popped > push_budget * 2) break;

    if (last_solution_cost >= 0 && (int)out.size() >= collect && cur.cost > last_solution_cost) break;

    if (cur.pos == (int)broken.size() && dfa.accept[(size_t)cur.state]) {
      out.push_back(RepairCandidate{cur.cost, materialize_path(steps, cur.path_id)});
      last_solution_cost = cur.cost;
      if ((int)out.size() >= collect) {
        // keep going until cost increases, then stop
      }
      continue;
    }

    // Delete: consume input without emitting.
    if (cur.pos < (int)broken.size()) {
      try_push(cur.cost + 1, cur.state, cur.pos + 1, cur.path_id);
    }

    // Insert or match/substitute: follow transitions and emit char.
    const auto& m = dfa.trans[(size_t)cur.state];
    for (const auto& kv : m) {
      char c = kv.first;
      int ns = kv.second;
      // Insert
      try_push(cur.cost + 1, ns, cur.pos, push_emit(cur.path_id, c));
      // Match/Substitute
      if (cur.pos < (int)broken.size()) {
        int add = (broken[(size_t)cur.pos] == c) ? 0 : 1;
        try_push(cur.cost + add, ns, cur.pos + 1, push_emit(cur.path_id, c));
      }
    }
  }

  // De-dup and cap.
  std::unordered_set<std::string> dedup_seen;
  std::vector<RepairCandidate> dedup;
  dedup.reserve(out.size());
  for (auto& c : out) {
    if (!dedup_seen.insert(c.text).second) continue;
    dedup.push_back(std::move(c));
    if ((int)dedup.size() >= want) break;
  }
  return dedup;
}

static int env_int_or(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

static DFA learn_with_learner(
    const Options& opt,
    const std::vector<std::string>& positives,
    const std::vector<std::string>& negatives,
    const Oracle& oracle) {
  std::string learner = to_lower(opt.learner);
  if (learner == "rpni") {
    RPNI l(positives, negatives);
    return l.learn();
  }
  if (learner == "rpni_xover") {
    int pairs = (opt.xover_pairs >= 0) ? opt.xover_pairs : env_int_or("LSTAR_RPNI_XOVER_PAIRS", 50);
    int checks = (opt.xover_checks >= 0) ? opt.xover_checks : env_int_or("LSTAR_RPNI_XOVER_CHECKS", 10);
    XoverRPNI l(
        positives,
        negatives,
        [&](std::string_view w) { return oracle.validate_text(w); },
        pairs,
        checks);
    return l.learn();
  }
  throw std::runtime_error("unknown --learner (expected rpni or rpni_xover): " + opt.learner);
}

static std::vector<std::vector<std::pair<char, int>>> build_adjacency(const DFA& dfa) {
  std::vector<std::vector<std::pair<char, int>>> adj(dfa.trans.size());
  for (size_t s = 0; s < dfa.trans.size(); s++) {
    adj[s].reserve(dfa.trans[s].size());
    for (const auto& kv : dfa.trans[s]) adj[s].push_back(kv);
  }
  return adj;
}

static std::optional<std::string> sample_accepted_string(
    const DFA& dfa,
    const std::vector<std::vector<std::pair<char, int>>>& adj,
    int length,
    std::mt19937_64& rng) {
  if (length < 0) return std::nullopt;
  if (length == 0) {
    if (dfa.accept[(size_t)dfa.start]) return std::string();
    return std::nullopt;
  }
  for (int attempt = 0; attempt < 60; attempt++) {
    int st = dfa.start;
    std::string out;
    out.reserve((size_t)length);
    bool dead = false;
    for (int i = 0; i < length; i++) {
      const auto& outs = adj[(size_t)st];
      if (outs.empty()) {
        dead = true;
        break;
      }
      std::uniform_int_distribution<size_t> dist(0, outs.size() - 1);
      auto [ch, ns] = outs[dist(rng)];
      out.push_back(ch);
      st = ns;
    }
    if (dead) continue;
    if (dfa.accept[(size_t)st]) return out;
  }
  return std::nullopt;
}

int main(int argc, char** argv) {
  try {
    auto parsed = parse_args(argc, argv);
    if (!parsed) return 0;
    Options opt = *parsed;

    std::vector<std::string> pos_lines = read_lines(*opt.positives);
    std::vector<std::string> train_positives = pos_lines;
    std::unordered_set<std::string> positives;
    positives.reserve(pos_lines.size() * 2);
    size_t min_len = SIZE_MAX;
    size_t max_len = 0;
    for (auto& s : pos_lines) {
      positives.insert(s);
      min_len = std::min(min_len, s.size());
      max_len = std::max(max_len, s.size());
    }

    std::vector<std::string> negatives;
    if (opt.negatives && fs::is_regular_file(*opt.negatives)) {
      negatives = read_lines(*opt.negatives);
    }
    std::cerr << "[INFO] Loaded positives=" << positives.size() << ", negatives=" << negatives.size() << "\n";

    Oracle oracle;
    oracle.repo_root = opt.repo_root;
    oracle.category = opt.category;
    oracle.timeout_ms = opt.oracle_timeout_ms;
    oracle.debug = getenv_bool("BETAMAX_DEBUG_ORACLE");
    oracle.verbose = opt.verbose;
    if (opt.oracle_validator) oracle.override_cmd = split_cmdline(*opt.oracle_validator);

    struct OracleStatsReporter {
      const Oracle& oracle;
      bool enabled = false;
      ~OracleStatsReporter() {
        if (!enabled) return;
        const auto& s = oracle.stats;
        std::cerr << "[INFO] Oracle stats: total=" << s.total << " correct=" << s.correct << " incorrect=" << s.incorrect
                  << " incomplete=" << s.incomplete << " seconds_total=" << s.seconds_total << "\n";
      }
    } oracle_stats_reporter{oracle, opt.verbose};

    if (opt.verbose) {
      std::cerr << "[DEBUG] Options: category=" << opt.category << " learner=" << opt.learner << " max_attempts=" << opt.max_attempts
                << " max_cost=" << opt.max_cost << " max_candidates=" << opt.max_candidates << " attempt_candidates=" << opt.attempt_candidates
                << " mutations=" << opt.mutations << " oracle_timeout_ms=" << opt.oracle_timeout_ms << "\n";
      if (opt.dfa_cache) std::cerr << "[DEBUG] DFA cache: " << opt.dfa_cache->string() << " init_cache=" << (opt.init_cache ? 1 : 0) << "\n";
      if (opt.oracle_validator) std::cerr << "[DEBUG] Oracle override cmd: " << *opt.oracle_validator << "\n";
    }

    // Mutation-based sample augmentation (like Python betaMax --mutations):
    // generate mutated strings from positives, label them with the oracle,
    // and add oracle-accepted ones to positives and rejected ones to negatives.
    if (opt.mutations > 0 && !train_positives.empty()) {
      std::vector<char> alphabet = derive_alphabet_from_positives(train_positives);
      uint64_t seed = 0;
      if (opt.mutations_seed) {
        seed = *opt.mutations_seed;
      } else if (opt.seed) {
        seed = *opt.seed;
      } else {
        seed = (uint64_t)std::random_device{}();
      }
      std::mt19937_64 rng(seed);

      std::unordered_set<std::string> neg_set(negatives.begin(), negatives.end());
      neg_set.reserve((size_t)negatives.size() * 2 + 8);

      std::vector<std::string> muts;
      if (opt.mutations_deterministic) {
        muts = generate_mutations_deterministic(train_positives, opt.mutations, alphabet);
      } else {
        muts = generate_mutations_random(train_positives, opt.mutations, opt.mutations_edits, alphabet, rng);
      }

      int accepted = 0;
      int rejected = 0;
      int skipped = 0;
      for (const auto& s : muts) {
        if (positives.count(s) || neg_set.count(s)) {
          skipped++;
          continue;
        }
        bool ok = oracle.validate_text(s);
        if (ok) {
          train_positives.push_back(s);
          positives.insert(s);
          min_len = std::min(min_len, s.size());
          max_len = std::max(max_len, s.size());
          accepted++;
        } else {
          negatives.push_back(s);
          neg_set.insert(s);
          rejected++;
        }
      }
      std::cerr << "[INFO] Mutation augmentation: requested=" << opt.mutations << ", generated=" << muts.size()
                << ", accepted=" << accepted << ", rejected=" << rejected << ", skipped=" << skipped
                << "; totals P=" << positives.size() << ", N=" << negatives.size() << "\n";
    }

    if (opt.init_cache) {
      auto t0 = std::chrono::steady_clock::now();
      DFA dfa = learn_with_learner(opt, train_positives, negatives, oracle);
      auto t1 = std::chrono::steady_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      size_t trans_count = 0;
      for (const auto& m : dfa.trans) trans_count += m.size();
      std::cerr << "[INFO] Learned DFA (precompute) states=" << dfa.trans.size() << ", transitions=" << trans_count << " in " << ms << "ms\n";

      // Optional oracle-guided sampling (disabled by default; can be expensive).
      if (!opt.eq_disable_sampling && opt.eq_max_rounds > 0 && opt.eq_max_oracle > 0 && opt.eq_samples_per_length > 0) {
        std::unordered_set<std::string> negative_set(negatives.begin(), negatives.end());
        std::mt19937_64 rng(opt.seed ? *opt.seed : (uint64_t)std::random_device{}());

        int oracle_calls = 0;
        size_t added_total_neg = 0;
        size_t added_total_pos = 0;
        for (int round = 0; round < opt.eq_max_rounds; round++) {
          auto adj = build_adjacency(dfa);
          size_t added_round_neg = 0;
          size_t added_round_pos = 0;

          for (int len = 0; len <= opt.eq_max_length; len++) {
            for (int s = 0; s < opt.eq_samples_per_length; s++) {
              if (oracle_calls >= opt.eq_max_oracle) break;
              auto sample = sample_accepted_string(dfa, adj, len, rng);
              if (!sample) continue;
              if (positives.count(*sample) || negative_set.count(*sample)) continue;

              oracle_calls++;
              bool ok = oracle.validate_text(*sample);
              if (ok) {
                train_positives.push_back(*sample);
                positives.insert(*sample);
                min_len = std::min(min_len, sample->size());
                max_len = std::max(max_len, sample->size());
                added_total_pos++;
                added_round_pos++;
              } else {
                negatives.push_back(*sample);
                negative_set.insert(*sample);
                added_round_neg++;
                added_total_neg++;
              }
            }
            if (oracle_calls >= opt.eq_max_oracle) break;
          }

          if (added_round_neg == 0 && added_round_pos == 0) break;
          dfa = learn_with_learner(opt, train_positives, negatives, oracle);
        }

        if (added_total_neg > 0 || added_total_pos > 0) {
          trans_count = 0;
          for (const auto& m : dfa.trans) trans_count += m.size();
          std::cerr << "[INFO] EQ sampling (precompute) added positives=" << added_total_pos << ", negatives=" << added_total_neg << ", oracle_calls=" << oracle_calls
                    << "; DFA now states=" << dfa.trans.size() << ", transitions=" << trans_count << "\n";
        }
      }

      std::cerr << "[INFO] Writing DFA cache: " << opt.dfa_cache->string() << "\n";
      if (!write_dfa_cache(*opt.dfa_cache, dfa)) {
        std::cerr << "[ERROR] Failed to write DFA cache: " << opt.dfa_cache->string() << "\n";
        return 1;
      }
      return 0;
    }

    std::string broken = opt.broken ? *opt.broken : read_file_all_strip_one_trailing_newline(*opt.broken_file);
    std::cerr << "[INFO] Broken len=" << broken.size() << "\n";

    // Active loop (like original betaMax): learn -> propose closest fix -> oracle check -> if reject, add as negative and relearn.
    std::unordered_set<std::string> neg_set(negatives.begin(), negatives.end());
    neg_set.reserve((size_t)negatives.size() * 2 + 8);

    int max_cost = opt.max_cost;
    bool unbounded_cost = (max_cost < 0);
    // Simple length guard to avoid absurd expansions.
    size_t hard_min = 0;
    size_t hard_max = std::numeric_limits<size_t>::max();
    if (!unbounded_cost) {
      hard_min = (min_len == SIZE_MAX) ? 0 : (min_len > (size_t)max_cost ? min_len - (size_t)max_cost : 0);
      hard_max = (max_len + (size_t)max_cost);
      if (broken.size() < hard_min || broken.size() > hard_max) {
        std::cerr << "[INFO] Broken length outside training-range±max_cost; search may need higher --max-cost.\n";
      }
    }

    DFA dfa;
    bool need_learn = true;
    bool loaded_cache = false;
    if (opt.dfa_cache && fs::is_regular_file(*opt.dfa_cache)) {
      auto loaded = read_dfa_cache(*opt.dfa_cache);
      if (loaded) {
        dfa = *loaded;
        need_learn = false;
        loaded_cache = true;
        size_t tc = 0;
        for (const auto& m : dfa.trans) tc += m.size();
        std::cerr << "[INFO] Loaded DFA cache: states=" << dfa.trans.size() << ", transitions=" << tc
                  << " from " << opt.dfa_cache->string() << "\n";
      } else {
        std::cerr << "[WARN] Failed to read DFA cache: " << opt.dfa_cache->string() << " (will relearn)\n";
      }
    }

    for (int attempt = 0; attempt < opt.max_attempts; attempt++) {
      std::cerr << "[ATTEMPT " << attempt << "] learner=" << opt.learner << " P=" << positives.size() << " N=" << negatives.size() << "\n";

      long long ms = 0;
      if (need_learn) {
        auto t0 = std::chrono::steady_clock::now();
        dfa = learn_with_learner(opt, train_positives, negatives, oracle);
        auto t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        loaded_cache = false;
      } else if (loaded_cache) {
        std::cerr << "[INFO] Using cached DFA (no relearn this attempt)\n";
      }

      size_t trans_count = 0;
      for (const auto& m : dfa.trans) trans_count += m.size();
      if (need_learn) {
        std::cerr << "[INFO] Learned DFA states=" << dfa.trans.size() << ", transitions=" << trans_count << " in " << ms << "ms\n";
      } else {
        std::cerr << "[INFO] DFA states=" << dfa.trans.size() << ", transitions=" << trans_count << "\n";
      }
      need_learn = false;

      // Optional oracle-guided sampling (disabled by default; can be expensive).
      if (!opt.eq_disable_sampling && opt.eq_max_rounds > 0 && opt.eq_max_oracle > 0 && opt.eq_samples_per_length > 0) {
        std::unordered_set<std::string> negative_set(negatives.begin(), negatives.end());
        std::mt19937_64 rng(opt.seed ? *opt.seed : (uint64_t)std::random_device{}());

        int oracle_calls = 0;
        size_t added_total_neg = 0;
        size_t added_total_pos = 0;
        for (int round = 0; round < opt.eq_max_rounds; round++) {
          auto adj = build_adjacency(dfa);
          size_t added_round_neg = 0;
          size_t added_round_pos = 0;

          for (int len = 0; len <= opt.eq_max_length; len++) {
            for (int s = 0; s < opt.eq_samples_per_length; s++) {
              if (oracle_calls >= opt.eq_max_oracle) break;
              auto sample = sample_accepted_string(dfa, adj, len, rng);
              if (!sample) continue;
              if (positives.count(*sample) || negative_set.count(*sample)) continue;

              oracle_calls++;
              bool ok = oracle.validate_text(*sample);
              if (ok) {
                train_positives.push_back(*sample);
                positives.insert(*sample);
                min_len = std::min(min_len, sample->size());
                max_len = std::max(max_len, sample->size());
                added_total_pos++;
                added_round_pos++;
              } else {
                negatives.push_back(*sample);
                negative_set.insert(*sample);
                neg_set.insert(*sample);
                added_round_neg++;
                added_total_neg++;
              }
            }
            if (oracle_calls >= opt.eq_max_oracle) break;
          }

          if (added_round_neg == 0 && added_round_pos == 0) break;
          dfa = learn_with_learner(opt, train_positives, negatives, oracle);
        }

        if (added_total_neg > 0 || added_total_pos > 0) {
          trans_count = 0;
          for (const auto& m : dfa.trans) trans_count += m.size();
          std::cerr << "[INFO] EQ sampling added positives=" << added_total_pos << ", negatives=" << added_total_neg << ", oracle_calls=" << oracle_calls
                    << "; DFA now states=" << dfa.trans.size() << ", transitions=" << trans_count << "\n";
        }
      }

      // Propose candidate repairs from DFA language near the broken input.
      int want = std::min(opt.max_candidates, opt.attempt_candidates);
      if (want < 1) want = 1;
      std::vector<RepairCandidate> candidates = unbounded_cost
          ? generate_candidates_to_dfa_unbounded(dfa, broken, want)
          : generate_candidates_to_dfa(dfa, broken, max_cost, want);
      std::cerr << "[INFO] DFA candidates (<=max_cost) count=" << candidates.size() << "\n";
      if (candidates.empty()) {
        if (unbounded_cost) {
          std::cerr << "[INFO] No candidates available (unbounded cost search, budget exhausted)\n";
        } else {
          std::cerr << "[INFO] No candidates available within --max-cost=" << max_cost << "\n";
        }
        return 2;
      }

      // Try candidates in order; accept on first oracle OK.
      std::optional<std::string> best_rejected;
      for (size_t i = 0; i < candidates.size(); i++) {
        const auto& c = candidates[i];
        if (opt.verbose) {
          std::cerr << "[DEBUG] Candidate " << (i + 1) << "/" << candidates.size() << " cost=" << c.cost << " len=" << c.text.size()
                    << " text='" << preview_for_log(c.text) << "'\n";
        }
        if (c.text.size() < hard_min || c.text.size() > hard_max) {
          if (opt.verbose) std::cerr << "[DEBUG] Candidate skipped: length guard\n";
          continue;
        }
        if (neg_set.count(c.text)) {
          if (opt.verbose) std::cerr << "[DEBUG] Candidate skipped: already negative\n";
          continue;
        }
        bool ok = oracle.validate_text(c.text);
        if (ok) {
          std::cout << c.text << "\n";
          if (opt.output_file) {
            std::ofstream out(*opt.output_file, std::ios::binary);
            out.write(c.text.data(), (std::streamsize)c.text.size());
          }
          std::cerr << "[INFO] Repaired with cost=" << c.cost << " after trying " << (i + 1) << " candidate(s)\n";
          return 0;
        }
        if (!best_rejected) best_rejected = c.text;
      }

      // Oracle rejected the closest candidate(s): add the closest rejected as a new negative and relearn.
      if (!best_rejected) {
        std::cerr << "[INFO] All candidate(s) are already in negatives or filtered; stopping.\n";
        return 2;
      }
      if (!neg_set.insert(*best_rejected).second) {
        std::cerr << "[INFO] Candidate already marked negative; stopping.\n";
        return 2;
      }
      negatives.push_back(*best_rejected);
      std::cerr << "[INFO] Oracle rejected; adding negative counterexample and relearning.\n";
      need_learn = true;
    }

    std::cerr << "[INFO] Exceeded --max-attempts=" << opt.max_attempts << " without finding an oracle-accepted repair.\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }
}
