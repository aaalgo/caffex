#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#ifndef CAFFEX_IMPL
#undef LOG
#undef LOG_IF
#endif
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace caffex {

using namespace caffe; 
using std::vector;
using std::string;
using boost::shared_ptr;

// The Caffe Extractor
// The extractor is initialized from a model directory that contains the following files:
//  - caffe.model: network model
//  - caffe.params: trained parameters
//  - caffe.mean: mean image
class Caffex {
    Net<float> net;
    int input_batch;
    int input_channels;
    int input_height;
    int input_width;
    int output_dim;
    cv::Mat mean;
    vector<shared_ptr<Blob<float>>> output_blobs;

    void wrapInputLayer(std::vector<cv::Mat> *);
    void extractOutputValues (float *, int n);
    void preprocess(cv::Mat const &, cv::Mat *);
    void preprocess(cv::Mat const &image, vector<cv::Mat> *channels) {
        preprocess(image, &channels->at(0));
    }
    void preprocess(vector<cv::Mat> const &images, vector<cv::Mat> *channels) {
        int off = 0;
        for (auto const &image: images) {
            preprocess(image, &channels->at(off));
            off += input_channels;
        }
    }
public:
    Caffex (const string& model_dir, unsigned batch = 1);
    void apply (cv::Mat const &, vector<float> *);
    void apply (vector<cv::Mat> const &, cv::Mat *);
    int dim () const { return output_dim; }
    cv::Size inputSize () const {
        return cv::Size(input_width, input_height);
    }
};

}

