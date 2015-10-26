// This program tries to find matches between local
// interesting points from two images using
// various methods.
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include "caffex.h"

using namespace std;
using namespace boost;

struct Job {
    int label;
    string path;
    vector<float> ft;
};

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string model_dir;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model,m", po::value(&model_dir), "model directory")
    ;

    po::positional_options_description p;
    p.add("model", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) {
        cerr << desc;
        return 1;
    }

    vector<Job> jobs;
    {
        Job job;
        for (;;) {
            cin >> job.label;
            string line;
            getline(cin, line);
            if (!cin) break;
            unsigned off = 0;
            while (off < line.size() && isspace(line[off])) ++off;
            if (off >= line.size()) break;
            job.path = line.substr(off);
            jobs.push_back(job);
        }
    }
    progress_display progress(jobs.size(), cerr);
#pragma omp parallel
    {
        caffex::Caffex ex(model_dir);
#pragma omp for schedule(dynamic,1)
        for (unsigned i = 0; i < jobs.size(); ++i) {
            auto &job = jobs[i];
            cv::Mat mat = cv::imread(job.path);
            if (mat.total() ==0) continue;
            ex.apply(mat, &job.ft);
#pragma omp critical
            ++progress;
        }
    }
    for (auto const &job: jobs) {
        cout << job.label;
        for (unsigned i = 0; i < job.ft.size(); ++i) {
            cout << ' ' << (i+1) << ':' << job.ft[i];
        }
        cout << endl;
    }

    return 0;
}

