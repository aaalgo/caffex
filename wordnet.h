#pragma once
#include <string>
#include <vector>
#include <fstream>

namespace caffex {
    using std::string;
    using std::vector;
    using std::ifstream;

    struct WordNetBase {
        struct Entry {
            string text;
            string wnid;
            vector<unsigned> parents;
            vector<unsigned> children;
        };
    };

    class WordNet: public WordNetBase, public vector<WordNetBase::Entry> {
    public:
        WordNet (string const &path) {
            value_type vt;
            ifstream is(path.c_str());
            int n;
            while (is >> vt.wnid >> n) {
                vt.parents.clear();
                for (unsigned i = 0; i < n; ++i) {
                    unsigned p;
                    is >> p;
                    vt.parents.push_back(p);
                }
                is.get();
                getline(is, vt.text);
                push_back(vt);
            }
            for (unsigned i = 0; i < size(); ++i) {
                for (unsigned p: at(i).parents) {
                    at(p).children.push_back(i);
                }
            }
        }
    };
}
