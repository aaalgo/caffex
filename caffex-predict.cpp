#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <xgboost_wrapper.h>
#include "caffex-xgboost.h"

using namespace std;
using namespace boost;

struct Job {
    string path;
    string barcode;
    float pred;
};

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string model_dir;
    unsigned batch;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model,m", po::value(&model_dir), "")
    ("batch,b", po::value(&batch)->default_value(32), "")
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
        job.pred = 0;
        while (cin >> job.barcode >> job.path) {
            jobs.push_back(job);
        }
    }
    progress_display progress(jobs.size(), cerr);

#if 0   // the code below are for testing
    if (batch == 1) {
#pragma omp parallel
        {
            caffex::CaffexBoost ex(model_dir);
            vector<float> ft;
#pragma omp for schedule(dynamic,1)
            for (unsigned i = 0; i < jobs.size(); ++i) {
                auto &job = jobs[i];
                cv::Mat image = cv::imread(job.path);
                if (image.total() ==0) continue;
                ex.apply(image, &ft);
                job.pred = ft[0];
#pragma omp critical
                ++progress;
            }
        }
    }
    else {
        caffex::CaffexBoost ex(model_dir, batch);
        unsigned off = 0;
        while (off < jobs.size()) {
            unsigned begin = off;
            unsigned end = begin + batch;
            if (end > jobs.size()) end = jobs.size();
            //
            vector<cv::Mat> images(end-begin);
            for (unsigned i = 0; i < images.size(); ++i) {
                images[i] = cv::imread(jobs[begin + i].path);
            }
            cv::Mat pred;
            ex.apply(images, &pred);
            for (unsigned i = 0; i < images.size(); ++i) {
                jobs[begin + i].pred = pred.ptr<float>(i)[0];
            }
            off = end;
            progress += end - begin;
        }
    }
#endif
    unsigned nbatch = (jobs.size() + batch -1) / batch;
#pragma omp parallel
    {
        caffex::CaffexBoost ex(model_dir, batch);
        cv::Mat pred;
#pragma omp for schedule(dynamic,1)
        for (unsigned i = 0; i < nbatch; ++i) {
            unsigned begin = i * batch;
            unsigned end = begin + batch;
            if (end > jobs.size()) end = jobs.size();
            vector<cv::Mat> images(end-begin);
            for (unsigned i = 0; i < images.size(); ++i) {
                images[i] = cv::imread(jobs[begin + i].path);
            }
            ex.apply(images, &pred);
            for (unsigned i = 0; i < images.size(); ++i) {
                jobs[begin + i].pred = pred.ptr<float>(i)[0];
            }
#pragma omp critical
            progress += end - begin;
        }
    }

    for (auto const &job: jobs) {
        cout << job.pred << '\t' << job.barcode << '\t' << job.path << endl;
    }

    return 0;
}

