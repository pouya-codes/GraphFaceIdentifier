/*
 * Classifier.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: ubuntu
 */

#include "Classifier.h"

Classifier::Classifier() {

}

int Classifier::LoadModel(const string& model_file, const string& trained_file) {


	Caffe::set_mode(Caffe::GPU);

	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1)<< "Network should have exactly one output.";

	return 0 ;

}


/* Return the top N predictions. */
Prediction Classifier::Classify(const float* features) {


	cv::gpu::GpuMat features_gpu(4096, 1, CV_32FC1,(float*) features);

	Blob<float>* input_layer = net_->input_blobs()[0];
	float* input_data = input_layer->mutable_gpu_data();
	cv::gpu::GpuMat channel(4096, 1, CV_32FC1, input_data);
	features_gpu.copyTo(channel) ;

	net_->Forward();

	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	std::vector<float> output = std::vector<float>(begin, end);
	std::vector<int> maxN = Argmax(output, output_layer->channels());
	int idx = maxN[0];
	Prediction result ;
	result.first = idx ;
	result.second = output[idx] ;
	return result;


}

std::vector<int> Classifier::Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(),
			PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}




