#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <math.h>
#include "Classifier.h"
#include "FeatureExtractorDeploy.h"
#include "NVX/tracking/tracking.hpp"
#include "NVX/tracking/tracking_with_features_info.hpp"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms/draw_abstract.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/dnn/gpu_data.h>
#include "Settings.h"

#ifndef FACE_DETECTON_GRAPH_H_
#define FACE_DETECTON_GRAPH_H_

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
        } \
    }
//        exit(1) /;
enum user_library_e {
	USER_LIBRARY_FACE_TRACKER = 1
};
enum user_kernel_e {
	USER_KERNEL_FACE_TRACKER    = VX_KERNEL_BASE(VX_ID_DEFAULT,
			USER_LIBRARY_FACE_TRACKER    ) + 0x001
};

struct Scalar {
	vx_uint8 values[3];
};
typedef std::vector<ObjectTrackerWithFeaturesInfo::TrackedObject*> TrackedObjectPointersVector;
struct _user_struct_event_tracker {
	bool isPressed;
	bool done;
	bool readNextFrame;
	bool pause;
	bool shouldShowFeatures;

	std::vector<int> labels;

	Scalar curColor;

	vx_coordinates2d_t objectTL;
	vx_coordinates2d_t objectBR;
	vx_coordinates2d_t currentPoint;

	ObjectTrackerWithFeaturesInfo* tracker;
	TrackedObjectPointersVector objects;
	std::vector<Scalar> colors;

	vx_uint32 frameWidth;
	vx_uint32 frameHeight;
	vx_image frame;

	_user_struct_event_tracker() :
			isPressed { false }, done { false }, readNextFrame { true }, pause {
					false }, shouldShowFeatures { false }, objectTL { 0, 0 }, objectBR {
					0, 0 }, currentPoint { 0, 0 }, tracker { NULL }, frameWidth {
					0 }, frameHeight { 0 }, frame { NULL } {
	}
} user_struct_event_tracker;
vx_enum VX_TYPE_USER_EVENT_TRACKER ;

static bool read_tracker_config_file(const std::string &nf, KeypointObjectTrackerParams &config,
		std::string &message) {
	std::unique_ptr<nvxio::ConfigParser> ftparser(nvxio::createConfigParser());
	ftparser->addParameter("pyr_levels",
			nvxio::OptionHandler::unsignedInteger(&config.pyr_levels,
					nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(8u)));
	ftparser->addParameter("lk_num_iters",
			nvxio::OptionHandler::unsignedInteger(&config.lk_num_iters,
					nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(100u)));
	ftparser->addParameter("lk_win_size",
			nvxio::OptionHandler::unsignedInteger(&config.lk_win_size,
					nvxio::ranges::atLeast(3u) & nvxio::ranges::atMost(32u)));
	ftparser->addParameter("max_corners",
			nvxio::OptionHandler::unsignedInteger(&config.max_corners,
					nvxio::ranges::atLeast(0u)));
	ftparser->addParameter("strength_threshold",
			nvxio::OptionHandler::real(&config.strength_threshold,
					nvxio::ranges::atLeast(1.f)
							& nvxio::ranges::atMost(254.f)));
	ftparser->addParameter("fast_type",
			nvxio::OptionHandler::unsignedInteger(&config.fast_type,
					nvxio::ranges::atLeast(9u) & nvxio::ranges::atMost(12u)));
	ftparser->addParameter("detector_cell_size",
			nvxio::OptionHandler::unsignedInteger(&config.detector_cell_size,
					nvxio::ranges::atLeast(1u) & nvxio::ranges::atMost(719u)));
	ftparser->addParameter("bb_decreasing_ratio",
			nvxio::OptionHandler::real(&config.bb_decreasing_ratio,
					nvxio::ranges::atLeast(0.f) & nvxio::ranges::atMost(1.f)));
	ftparser->addParameter("max_corners_in_cell",
			nvxio::OptionHandler::unsignedInteger(&config.max_corners_in_cell,
					nvxio::ranges::atLeast(0u)));
	ftparser->addParameter("x_num_of_cells",
			nvxio::OptionHandler::unsignedInteger(&config.x_num_of_cells,
					nvxio::ranges::atLeast(0u)));
	ftparser->addParameter("y_num_of_cells",
			nvxio::OptionHandler::unsignedInteger(&config.y_num_of_cells,
					nvxio::ranges::atLeast(0u)));

	message = ftparser->parse(nf);

	return message.empty();
}
static bool rect_intersect_checker(vx_rectangle_t t1,vx_rectangle_t t2, int overlap) {
	int t1_start_x = int(t1.start_x);
	int t1_start_y = int(t1.start_y);
	int t1_end_x = int(t1.end_x);
	int t1_end_y = int(t1.end_y);

	int t2_start_x = int(t2.start_x);
	int t2_start_y = int(t2.start_y);
	int t2_end_x = int(t2.end_x);
	int t2_end_y = int(t2.end_y);
	int pad = 0 ;
	if ( (t1_start_x - pad < t2_start_x  && t1_end_x + pad >  t2_end_x &&
		  t1_start_y - pad < t2_start_y  && t1_end_y + pad >  t2_end_y) or
		 (t2_start_x - pad <= t1_start_x && t2_end_x + pad >= t1_end_x &&
		  t2_start_y - pad <= t1_start_y && t2_end_y + pad >= t1_end_y))
		return true ;
//	if (t1.end_x < t2.start_x or t2.end_x < t1.start_x or t1.end_y < t2.start_y or t2.end_y < t2.start_y)



	int border = int(MAX(abs((t1_start_x- t1_end_x))/overlap , abs( t2_start_x- t1_end_x)/overlap)) ;
	if (abs(t1_start_x -t2_start_x) < border || abs(t1_end_x -t2_end_x) < border ||
		abs(t1_start_y -t2_start_y) < border || abs(t1_end_y -t2_end_y) < border ||
		(t1_start_x < t2_start_x && t1_end_x > t2_end_y && t1_start_y < t2_start_y && t1_end_y > t2_end_y) ||
		(t2_start_x < t1_start_x && t2_end_x > t1_end_y && t2_start_y < t1_start_y && t2_end_y > t1_end_y)
		)
		return true ;

	return false ;
}
double calcBlurriness( const cv::gpu::GpuMat &src );
cv::gpu::CascadeClassifier_GPU cascade_gpu ;
FeatureExtractorDeploy feature_extractor;
Classifier classifier;
dlib::frontal_face_detector detector ;
CvANN_MLP mlp;
float mins_mlp [3]= {0,0,100};
float maxs_mlp [3]= {1.00 ,0.1 ,350};

//********************************** Face Tracker *************************************************************
vx_node userFaceTracker(vx_graph graph,
						  vx_image input,
						  vx_image input_rgb,
						  vx_array face_tracker_array
						  ) {
	vx_context context = vxGetContext((vx_reference) graph);
	vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_FACE_TRACKER);
	ERROR_CHECK_OBJECT(kernel);
	vx_node node = vxCreateGenericNode(graph, kernel);
	ERROR_CHECK_OBJECT(node);
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 0, (vx_reference ) input));
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 1, (vx_reference ) input_rgb));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 2, (vx_reference ) face_tracker_array));
	ERROR_CHECK_STATUS( vxReleaseKernel(&kernel));

	return node;
}

vx_status VX_CALLBACK face_tracker_validator(vx_node node,
		const vx_reference parameters[], vx_uint32 num,
		vx_meta_format metas[]) {

	if (num == 3) {
		// parameters #1
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(
				vxQueryImage((vx_image )parameters[0], VX_IMAGE_FORMAT, &format,
						sizeof(format)));

		if (format != VX_DF_IMAGE_U8) {
			return VX_ERROR_INVALID_FORMAT;
		}

		// parameters #2
		format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(
				vxQueryImage((vx_image )parameters[1], VX_IMAGE_FORMAT, &format,
						sizeof(format)));

		if (format != VX_DF_IMAGE_RGB) {
			return VX_ERROR_INVALID_FORMAT;
		}

        // parameters #3 -- array of type VX_TYPE_USER_EVENT_TRACKER
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(
				vxQueryArray((vx_array )parameters[2], VX_ARRAY_ITEMTYPE, &type,
						sizeof(type)));

		vx_size num_item ;
        ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[2], VX_ARRAY_NUMITEMS, &num_item, sizeof( num_item ) ) );

		if (type != VX_TYPE_USER_EVENT_TRACKER || num_item!=1 ) {
            return VX_ERROR_INVALID_FORMAT;
		}

		// parameters #3 -- array of type VX_TYPE_USER_EVENT_TRACKER
		vx_size item_num = 1;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[2], VX_ARRAY_CAPACITY, &item_num,
						sizeof(item_num)));
		vx_enum object_type = VX_TYPE_USER_EVENT_TRACKER;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE,
						&object_type, sizeof(object_type)));

	} else {
		// invalid parameter
		return VX_ERROR_INVALID_PARAMETERS;
	}
	return VX_SUCCESS;
}

vx_status VX_CALLBACK face_tracker_init(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	cascade_gpu.load(CASCADE_FILE_PATH) ;
	detector = dlib::get_frontal_face_detector();
	feature_extractor.LoadModel(NET_PATH_VGG, NET_WEIGHTS_VGG) ;
	classifier.LoadModel(NET_PATH_CLASSIFIER,NET_WEIGHTS_CLASSIFIER) ;
	mlp.load(MLP_PATH) ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK face_tracker_deinit(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	cascade_gpu.release() ;
	classifier.~Classifier() ;
	mlp.~CvANN_MLP() ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK face_tracker_process(vx_node node,
		const vx_reference * refs, vx_uint32 num) {
	vx_image input_gray = (vx_image) refs[0];
	vx_image input_rgb = (vx_image) refs[1];
	vx_array face_tracker_array = (vx_array) refs[2];

	vx_uint32 width_gray = 0, height_gray = 0;
	ERROR_CHECK_STATUS(	vxQueryImage(input_gray, VX_IMAGE_WIDTH, &width_gray, sizeof(width_gray)));
	ERROR_CHECK_STATUS(	vxQueryImage(input_gray, VX_IMAGE_HEIGHT, &height_gray, sizeof(height_gray)));

	vx_rectangle_t rect = { 0, 0, width_gray, height_gray };
	vx_imagepatch_addressing_t addr_input = { 0 }, addr_output = { 0 };
	void * ptr_input = NULL, *ptr_output = NULL;
	vx_map_id map_id_input, map_id_output;
	ERROR_CHECK_STATUS(	vxMapImagePatch(input_gray, &rect, 0, &map_id_input, &addr_input,
					&ptr_input, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0));

	cv::gpu::GpuMat mat_input(height_gray, width_gray, CV_8U, ptr_input, addr_input.stride_y);
	ERROR_CHECK_STATUS(vxUnmapImagePatch(input_gray, map_id_input));

	cv::gpu::GpuMat objbuf;
	int detections_number = cascade_gpu.detectMultiScale(mat_input, objbuf, 1.1, 4);
	cv::Mat obj_host;
	objbuf.colRange(0, detections_number).download(obj_host);
	cv::Rect* faces = obj_host.ptr<cv::Rect>();

	if (detections_number > 0) {
        void* ptr_tracker ;
        vx_size stride_tracker = sizeof(_user_struct_event_tracker);
        vx_map_id map_id_tracker ;
        ERROR_CHECK_STATUS(vxMapArrayRange(face_tracker_array,0,1,&map_id_tracker,&stride_tracker,&ptr_tracker,VX_READ_AND_WRITE,VX_MEMORY_TYPE_HOST,0)) ;
        _user_struct_event_tracker face_tracker = vxArrayItem(_user_struct_event_tracker, ptr_tracker, 0,	stride_tracker);


    	vx_uint32 width_rgb = 0, height_rgb = 0;
    	ERROR_CHECK_STATUS(	vxQueryImage(input_rgb, VX_IMAGE_WIDTH, &width_rgb, sizeof(width_rgb)));
    	ERROR_CHECK_STATUS(	vxQueryImage(input_rgb, VX_IMAGE_HEIGHT, &height_rgb, sizeof(height_rgb)));

		vx_rectangle_t rect_frame = { 0, 0,
				(vx_uint32) width_rgb , (vx_uint32) height_rgb };

		vx_imagepatch_addressing_t addr_input_ = { 0 };
		void * ptr_input_ = NULL;
		vx_map_id map_id_input_;
		ERROR_CHECK_STATUS(	vxMapImagePatch(input_rgb, &rect_frame, 0,
						&map_id_input_, &addr_input_, &ptr_input_,
						VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0));
		cv::gpu::GpuMat frame_gpu(height_rgb, width_rgb,
				CV_8UC3, ptr_input_, addr_input_.stride_y);
		ERROR_CHECK_STATUS(
				vxUnmapImagePatch(input_rgb, map_id_input_));
		for (int i = 0; i < detections_number; i++) {
//			vx_int32 pading = IMAGE_PAD_PERSENT * faces[i].width;
//			vx_rectangle_t face_rect = {
//				(vx_uint32) faces[i].x      * RESIZE_LEVEL -  pading ,
//				(vx_uint32) faces[i].y      * RESIZE_LEVEL -  pading ,
//				(vx_uint32) faces[i].width  * RESIZE_LEVEL + ( 2 * pading),
//				(vx_uint32) faces[i].height * RESIZE_LEVEL + ( 2 * pading)
//			} ;




			vx_rectangle_t face_rect = { (vx_uint32) faces[i].x	* RESIZE_LEVEL_DEPLOY,
										 (vx_uint32) faces[i].y	* RESIZE_LEVEL_DEPLOY,
										 (vx_uint32) faces[i].x	* RESIZE_LEVEL_DEPLOY + (vx_uint32) faces[i].width  * RESIZE_LEVEL_DEPLOY,
										 (vx_uint32) faces[i].y * RESIZE_LEVEL_DEPLOY + (vx_uint32) faces[i].height * RESIZE_LEVEL_DEPLOY };
				bool new_face = true ;
				auto colorIt = face_tracker.colors.begin();
				auto lableIt = face_tracker.labels.begin();
				for (auto it = face_tracker.objects.begin();
						it != face_tracker.objects.end();) {
					ObjectTrackerWithFeaturesInfo::TrackedObject* obj = *it;
					ObjectTracker::ObjectStatus status = obj->getStatus();
					vx_rectangle_t rect = obj->getLocation();
					if (status != ObjectTracker::LOST && rect_intersect_checker(rect, face_rect, 8)) {

					if (*lableIt < 0) {

						cv::Rect mat_rect_rgb = cv::Rect(
								faces[i].x * RESIZE_LEVEL_DEPLOY,
								faces[i].y * RESIZE_LEVEL_DEPLOY,
								faces[i].width * RESIZE_LEVEL_DEPLOY,
								faces[i].height * RESIZE_LEVEL_DEPLOY);
						if (mat_rect_rgb.x > 0 && mat_rect_rgb.y > 0
								&& mat_rect_rgb.width > 0
								&& mat_rect_rgb.height > 0) {
							cv::gpu::GpuMat mat_face = frame_gpu(mat_rect_rgb);

							cv::gpu::GpuMat img_gpu_ycrcb, resized;
							cv::gpu::resize(mat_face, resized,
									cv::Size(224, 224));
							cv::gpu::cvtColor(resized, img_gpu_ycrcb,
									CV_BGR2YCrCb);
							std::vector<cv::gpu::GpuMat> channels;
							cv::gpu::split(img_gpu_ycrcb, channels);
							cv::gpu::equalizeHist(channels[0], channels[0]);
							merge(channels, img_gpu_ycrcb);
							cv::gpu::GpuMat output;
							cv::gpu::cvtColor(img_gpu_ycrcb, output,
									CV_YCrCb2BGR);

							const float* features =
									feature_extractor.GetFeatures(output);
							Prediction p = classifier.Classify(features);

							float probibility = p.second;
							float blur = calcBlurriness(mat_face);
							int dimension = (int) mat_rect_rgb.width;
							cv::Mat response(1, 1, CV_32FC1);
							cv::Mat sample(1, 3, CV_32FC1);
				            sample.at<float>  (0) = (probibility  - mins_mlp[0]) / ( maxs_mlp[0] - mins_mlp[0] ) ;
				            sample.at<float>  (1) = (blur    - mins_mlp[1]) / ( maxs_mlp[1] - mins_mlp[1] );
				            sample.at<float>  (2) = (dimension   - mins_mlp[2]) / ( maxs_mlp[2] - mins_mlp[2] );

							mlp.predict(sample, response);
							float reliable_value = response.at<float>(0, 0);
							bool reliable = false;
							if (reliable_value >= 0.0 || blur<0.0007)
								reliable = true;

							if (reliable) {
								if(probibility > 0.40){
									*lableIt = p.first;
									*colorIt= { {0, 255, 0}};
								}
								else {
									*lableIt = -1;
									*colorIt= { {255, 0, 0}};
								}

							}
							}
						}
						new_face=false ;
						break ;

					}
					++it;
					++colorIt;
					++lableIt;

				}

	            if(new_face) {


					cv::Rect mat_rect_rgb = cv::Rect(faces[i].x * RESIZE_LEVEL_DEPLOY,
							faces[i].y * RESIZE_LEVEL_DEPLOY,
							faces[i].width * RESIZE_LEVEL_DEPLOY,
							faces[i].height * RESIZE_LEVEL_DEPLOY);

	            	cv::gpu::GpuMat face_pixels = frame_gpu(mat_rect_rgb) ;

	            	//****************dlib

	            	cv::gpu::GpuMat face_pixels_gray ;

	            	cv::gpu::cvtColor(face_pixels,face_pixels_gray,cv::COLOR_BGR2GRAY) ;

//					face_detection_dlib_timer.tic() ;
			        cv::Mat face_pixels_gray_cut ;
			        face_pixels_gray.download(face_pixels_gray_cut) ;
					dlib::array2d<unsigned char> img_dlib;
					dlib::assign_image(img_dlib, dlib::cv_image<unsigned char>(face_pixels_gray_cut));
					std::vector<dlib::rectangle> dets = detector(img_dlib);
//					time_face_detection_dlib = face_detection_dlib_timer.toc() ;

	            	//****************dlib
//
					if (dets.size() == 0)
						break ;
//								std::cout << "Number of faces detected by dlib : " << dets.size() << std::endl;

//			            	std::cout << "new face" <<std::endl ;

	            	face_tracker.objects.push_back(
	            			face_tracker.tracker->addObject(face_rect));

					cv::gpu::GpuMat img_gpu_ycrcb,resized ;
					cv::gpu::resize(face_pixels, resized, cv::Size(224, 224));
					cv::gpu::cvtColor(resized, img_gpu_ycrcb, CV_BGR2YCrCb);
					std::vector<cv::gpu::GpuMat> channels;
					cv::gpu::split(img_gpu_ycrcb, channels);
					cv::gpu::equalizeHist(channels[0], channels[0]);
					merge(channels, img_gpu_ycrcb);
					cv::gpu::GpuMat output;
					cv::gpu::cvtColor(img_gpu_ycrcb, output, CV_YCrCb2BGR);

					const float* features = feature_extractor.GetFeatures(output);
					Prediction p =classifier.Classify(features) ;

		        	float probibility =  p.second ;
		        	float blur =  calcBlurriness(face_pixels) ;
		        	int dimension     =  (int) mat_rect_rgb.width  ;
		            cv::Mat response(1, 1, CV_32FC1);
		            cv::Mat sample (1, 3, CV_32FC1);
		            sample.at<float>  (0) = (probibility  - mins_mlp[0]) / ( maxs_mlp[0] - mins_mlp[0] ) ;
		            sample.at<float>  (1) = (blur    - mins_mlp[1]) / ( maxs_mlp[1] - mins_mlp[1] );
		            sample.at<float>  (2) = (dimension   - mins_mlp[2]) / ( maxs_mlp[2] - mins_mlp[2] );

		            mlp.predict(sample, response);
		            float reliable_value = response.at<float>(0,0);
		            bool reliable = false ;
		            if(reliable_value >= 0.0 || blur<0.0007)
		                reliable = true ;

					if(reliable) {
						if(probibility>0.5 ) {
							face_tracker.labels.push_back(p.first) ;
							face_tracker.colors.push_back( { { 0, 255, 0 } });
						}

						else
						{
							face_tracker.labels.push_back(-1) ;
							face_tracker.colors.push_back( { { 255, 0, 0 } });

						}
					}
					else {
						face_tracker.labels.push_back(-2);
						face_tracker.colors.push_back( { { 0, 0, 255 } });
					}

					std::cout << face_tracker.labels.size()<<"--"<<p.first << "-" << p.second <<"-"<<blur<<std::endl ;


	            }


		}
		vxArrayItem(_user_struct_event_tracker, ptr_tracker, 0,	stride_tracker) =face_tracker ;
		ERROR_CHECK_STATUS(vxUnmapArrayRange(face_tracker_array,map_id_tracker)) ;
	}


	return VX_SUCCESS;
}



double calcBlurriness( const cv::gpu::GpuMat &src )
{
	cv::gpu::GpuMat Gx, Gy;
	cv::gpu::Sobel( src, Gx, CV_32F, 1, 0 );
	cv::gpu::Sobel( src, Gy, CV_32F, 0, 1 );
    double normGx = norm( Gx );
    double normGy = norm( Gy );
    double sumSq = normGx * normGx + normGy * normGy;
    return static_cast<float>( 1. / ( sumSq / src.size().area() + 1e-6 ));
}


#endif /* FACE_DETECTON_GRAPH_H_ */
