/*
 * Classifier.h
 *
 *  Created on: Dec 6, 2016
 *      Author: ubuntu
 */

#ifndef FEATUREEXTRACTORDEPLOY_H_
#define FEATUREEXTRACTORDEPLOY_H_
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
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
class FeatureExtractorDeploy {
 public:
	FeatureExtractorDeploy();

	const float* GetFeatures(const cv::gpu::GpuMat& img);
	bool LoadModel(const string& model_file, const string& trained_file);

 private:

  const float* Predict(const cv::gpu::GpuMat& img);

  void WrapInputLayer(std::vector<cv::gpu::GpuMat>* input_channels);

  void Preprocess(const cv::gpu::GpuMat& img,
                  std::vector<cv::gpu::GpuMat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::gpu::GpuMat mean_;

};

#endif /* FEATUREEXTRACTORDEPLOY_H_ */
