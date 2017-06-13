/*
 * Classifier.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: ubuntu
 */

#include "FeatureExtractorDeploy.h"

FeatureExtractorDeploy::FeatureExtractorDeploy() {

}

bool FeatureExtractorDeploy::LoadModel(const string& model_file, const string& trained_file) {
	Caffe::set_mode(Caffe::GPU);

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1)<< "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
																<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());


	cv::Scalar channel_mean(129, 105, 94, 0);
	mean_ = cv::gpu::GpuMat(input_geometry_, CV_32FC3, channel_mean);



	Blob<float>* output_layer = net_->output_blobs()[0];

	return true ;
}

const float* FeatureExtractorDeploy::GetFeatures(const cv::gpu::GpuMat& img) {

	return Predict(img);
}

/* Load the mean file in binaryproto format. */


const float* FeatureExtractorDeploy::Predict(const cv::gpu::GpuMat& img) {

	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height,
			input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::gpu::GpuMat> input_channels;

	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	const boost::shared_ptr<caffe::Blob<float> > feature_blob =
			net_->blob_by_name("fc7");
	int batch_size = feature_blob->num();
	int dim_features = feature_blob->count() / batch_size;
	const float* feature_blob_data;

	feature_blob_data = feature_blob->gpu_data() + feature_blob->offset(0);

	return feature_blob_data;

}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FeatureExtractorDeploy::WrapInputLayer(
		std::vector<cv::gpu::GpuMat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::gpu::GpuMat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void FeatureExtractorDeploy::Preprocess(const cv::gpu::GpuMat& img,
		std::vector<cv::gpu::GpuMat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::gpu::GpuMat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::gpu::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::gpu::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::gpu::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::gpu::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::gpu::GpuMat sample_resized;
	if (sample.size() != input_geometry_)
		cv::gpu::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::gpu::GpuMat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::gpu::GpuMat sample_normalized;
	cv::gpu::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::gpu::split(sample_normalized, *input_channels);
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
															<< "Input channels are not wrapping the input layer of the network.";
}
