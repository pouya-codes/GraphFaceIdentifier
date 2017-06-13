/*
 * Classifier.h
 *
 *  Created on: Dec 6, 2016
 *      Author: ubuntu
 */

#ifndef POSEESTIMATOR_H_
#define POSEESTIMATOR_H_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
using namespace caffe;
using std::string;

static bool PairCompare(const std::pair<float, int>& lhs,
		const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}
typedef std::pair<int, float> Prediction;

class Classifier {
public:
	Classifier();
	int LoadModel (const string& model_file, const string& trained_file) ;

	Prediction Classify(const float* features);

private:
	std::vector<int> Argmax(const std::vector<float>& v, int N) ;
	shared_ptr<Net<float> > net_;

};

#endif /* POSEESTIMATOR_H_ */
