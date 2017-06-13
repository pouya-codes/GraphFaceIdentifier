#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <math.h>
#include "FeatureExtractorDeploy.h"
#include "Classifier.h"
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
	USER_LIBRARY_FACE_DETECTION = 1, USER_LIBRARY_FACE_IDENTICATION = 2, USER_LIBRARY_RESULT_VALIDATOR = 3,
};
enum user_kernel_e {
	USER_KERNEL_FACE_DETECTION    = VX_KERNEL_BASE(VX_ID_DEFAULT,
			USER_LIBRARY_FACE_DETECTION    ) + 0x001,
	USER_KERNEL_FACE_IDENTICATION = VX_KERNEL_BASE(VX_ID_DEFAULT,
			USER_LIBRARY_FACE_IDENTICATION ) + 0x001,
	USER_KERNEL_RESULT_VALIDATOR  = VX_KERNEL_BASE(VX_ID_DEFAULT,
			USER_LIBRARY_RESULT_VALIDATOR  ) + 0x001,
};

typedef struct _user_struct_identication_result {
    vx_uint32  class_label ;         /*!< \brief The label of the predicted classes. */
    vx_float32 class_probability ;   /*!< \brief The probability of predicted classes. */
    vx_float32 face_blurriness  ;	 /*!< \brief The Blurriness of the image. */
} user_struct_identication_result;
vx_enum VX_TYPE_USER_IDENTICATION_RESULT ;


typedef struct _user_struct_validation_result {
    vx_uint32  start_x;          	 /*!< \brief The Start X coordinate. */
    vx_uint32  start_y;          	 /*!< \brief The Start Y coordinate. */
    vx_uint32  end_x;            	 /*!< \brief The End X coordinate. */
    vx_uint32  end_y;                /*!< \brief The End Y coordinate. */
    vx_uint32  class_label ;         /*!< \brief The label of the predicted classes. */
    vx_float32 class_probability ;   /*!< \brief The probability of predicted classes. */
    vx_float32 face_blurriness  ;	 /*!< \brief The Blurriness of the face. */
    vx_bool    validated ;			 /*!< \brief The Status of validation of the face. */
} user_struct_validation_result;
vx_enum VX_TYPE_USER_VALIDATION_RESULT ;

double calcBlurriness( const cv::gpu::GpuMat &src );
cv::gpu::CascadeClassifier_GPU cascade_gpu ;
FeatureExtractorDeploy feature_extractor;
Classifier classifier;
CvANN_MLP mlp;

float mins_mlp [3]= {0,0,100};
float maxs_mlp [3]= {1.00 ,0.1 ,350};

//********************************** Face Detection *************************************************************
vx_node userFaceDetection(vx_graph graph,
						  vx_image input,
						  vx_scalar detected_face_number,
						  vx_array face_areas
						  ) {
	vx_context context = vxGetContext((vx_reference) graph);
	vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_FACE_DETECTION);
	ERROR_CHECK_OBJECT(kernel);
	vx_node node = vxCreateGenericNode(graph, kernel);
	ERROR_CHECK_OBJECT(node);
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 0, (vx_reference ) input));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 1, (vx_reference ) detected_face_number));
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 2, (vx_reference ) face_areas));
	ERROR_CHECK_STATUS( vxReleaseKernel(&kernel));

	return node;
}

vx_status VX_CALLBACK face_detection_validator(vx_node node,
		const vx_reference parameters[], vx_uint32 num,
		vx_meta_format metas[]) {

	if (num == 3) {
		// the first parameter
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(
				vxQueryImage((vx_image )parameters[0], VX_IMAGE_FORMAT, &format,
						sizeof(format)));

		if (format != VX_DF_IMAGE_U8) {
			return VX_ERROR_INVALID_FORMAT;
		}
		// the second parameter


        vx_enum scaler_type = VX_TYPE_UINT8;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[1], VX_SCALAR_TYPE,
						&scaler_type, sizeof(scaler_type)));

		//the third parameter
		vx_size item_num = MAX_FACE_DETECT;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[2], VX_ARRAY_CAPACITY, &item_num,
						sizeof(item_num)));

		vx_enum object_type = VX_TYPE_RECTANGLE;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE,
						&object_type, sizeof(object_type)));
	} else {
		// invalid parameter
		return VX_ERROR_INVALID_PARAMETERS;
	}
	return VX_SUCCESS;
}

vx_status VX_CALLBACK face_detection_init(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	cascade_gpu.load(CASCADE_FILE_PATH) ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK face_detection_deinit(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	cascade_gpu.release() ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK face_detection_process(vx_node node,
		const vx_reference * refs, vx_uint32 num) {

	vx_image input = (vx_image) refs[0];
	vx_scalar detected_face = (vx_scalar) refs[1];
	vx_array face_areas = (vx_array) refs[2];

	vx_uint32 width = 0, height = 0;
	ERROR_CHECK_STATUS(	vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(	vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height)));

	vx_rectangle_t rect = { 0, 0, width, height };
	vx_imagepatch_addressing_t addr_input = { 0 }, addr_output = { 0 };
	void * ptr_input = NULL, *ptr_output = NULL;
	vx_map_id map_id_input, map_id_output;
	ERROR_CHECK_STATUS(	vxMapImagePatch(input, &rect, 0, &map_id_input, &addr_input,
					&ptr_input, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0));

	cv::gpu::GpuMat mat_input(height, width, CV_8U, ptr_input,	addr_input.stride_y);
	ERROR_CHECK_STATUS(vxUnmapImagePatch(input, map_id_input));

	cv::gpu::GpuMat objbuf;
	int detections_number = cascade_gpu.detectMultiScale(mat_input, objbuf, 1.1, 4);
	cv::Mat obj_host;
	objbuf.colRange(0, detections_number).download(obj_host);
	cv::Rect* faces = obj_host.ptr<cv::Rect>();

	if (detections_number > 0) {
		for (int i = 0; i < detections_number; i++) {
			vx_int32 pading = IMAGE_PAD_PERSENT_DEPLOY * (faces[i].width * RESIZE_LEVEL_DEPLOY);
			vx_rectangle_t face_rect = {
				(vx_uint32) faces[i].x      * RESIZE_LEVEL_DEPLOY -  pading ,
				(vx_uint32) faces[i].y      * RESIZE_LEVEL_DEPLOY -  pading ,
				(vx_uint32) faces[i].width  * RESIZE_LEVEL_DEPLOY + ( 2 * pading),
				(vx_uint32) faces[i].height * RESIZE_LEVEL_DEPLOY + ( 2 * pading)
			} ;
			if(face_rect.start_x<=0 || face_rect.start_y<=0 || (face_rect.start_x+ face_rect.end_x >(width*RESIZE_LEVEL_DEPLOY))||(face_rect.start_y+face_rect.end_y>(height*RESIZE_LEVEL_DEPLOY))) {
				detections_number-- ;
				continue ;
			}

			ERROR_CHECK_STATUS(vxAddArrayItems(face_areas,1,&face_rect,sizeof(vx_rectangle_t))) ;
		}
	}
    vx_uint8  ksize   = detections_number;
    ERROR_CHECK_STATUS( vxWriteScalarValue( detected_face, &ksize ) );

	return VX_SUCCESS;
}
//********************************** Face Identication *************************************************************
vx_node userFaceIdentication(vx_graph  graph,
							 vx_image  input,
							 vx_scalar detected_face_number,
							 vx_array  face_areas,
							 vx_array  identication_results) {

	vx_context context = vxGetContext((vx_reference) graph);
	vx_kernel kernel = vxGetKernelByEnum(context,
			USER_KERNEL_FACE_IDENTICATION);
	ERROR_CHECK_OBJECT(kernel);
	vx_node node = vxCreateGenericNode(graph, kernel);
	ERROR_CHECK_OBJECT(node);

	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 0, (vx_reference ) input));
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 1, (vx_reference ) detected_face_number));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 2, (vx_reference ) face_areas));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 3, (vx_reference ) identication_results));

	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return node;
}

vx_status VX_CALLBACK face_identication_validator(vx_node node,
		const vx_reference parameters[], vx_uint32 num,
		vx_meta_format metas[]) {

	if (num == 4) {

		// parameters #1 -- image of type VX_DF_IMAGE_RGB
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(
				vxQueryImage((vx_image )parameters[0], VX_IMAGE_FORMAT, &format,
						sizeof(format)));

		if (format != VX_DF_IMAGE_RGB) {
			return VX_ERROR_INVALID_FORMAT;
		}

        // parameters #2 -- scalar of type VX_TYPE_UINT8
        vx_enum type = VX_TYPE_INVALID;
        ERROR_CHECK_STATUS( vxQueryScalar( ( vx_scalar )parameters[1], VX_SCALAR_TYPE, &type, sizeof( type ) ) );

        if( type != VX_TYPE_UINT8 )
        {
            return VX_ERROR_INVALID_TYPE;
        }

        // parameters #3 -- array of type VX_TYPE_RECTANGLE
		ERROR_CHECK_STATUS(
				vxQueryArray((vx_array )parameters[2], VX_ARRAY_ITEMTYPE, &type,
						sizeof(type)));

		vx_size num_item ;
        ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[2], VX_ARRAY_NUMITEMS, &num_item, sizeof( num_item ) ) );

		if (type != VX_TYPE_RECTANGLE  ) {
            return VX_ERROR_INVALID_FORMAT;
		}

		// parameters #4 -- array of type VX_TYPE_USER_STRUCT
		vx_size item_num = MAX_FACE_DETECT;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[3], VX_ARRAY_CAPACITY, &item_num,
						sizeof(item_num)));
		vx_enum object_type = VX_TYPE_USER_IDENTICATION_RESULT;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[3], VX_ARRAY_ITEMTYPE,
						&object_type, sizeof(object_type)));


	} else {
		// invalid input parameter
		return VX_ERROR_INVALID_PARAMETERS;
	}

	return VX_SUCCESS;
}

vx_status VX_CALLBACK face_identication_init(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	feature_extractor.LoadModel(NET_PATH_VGG, NET_WEIGHTS_VGG) ;
	classifier.LoadModel(NET_PATH_CLASSIFIER,NET_WEIGHTS_CLASSIFIER) ;
	return VX_SUCCESS;
}

vx_status VX_CALLBACK face_identication_deinit(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {

	return VX_SUCCESS;
}

vx_status VX_CALLBACK face_identication_process(vx_node node,
		const vx_reference * refs, vx_uint32 num) {

	vx_image input =  (vx_image) refs[0];
	vx_scalar detected_face_number = (vx_scalar) refs[1];
	vx_array face_areas = (vx_array) refs[2];
	vx_array identication_results = (vx_array) refs[3];

    vx_uint8  ksize   = 0;
    ERROR_CHECK_STATUS( vxReadScalarValue( detected_face_number, &ksize ) );

	int detected_faces = (int) ksize ;


	if (detected_faces > 0) {
		//access to detected_faces array
		vx_map_id map_id_input;
	    vx_size stride = sizeof(vx_rectangle_t);
		vx_map_id map_id ;
		void *base = NULL;
		ERROR_CHECK_STATUS(
				vxMapArrayRange(face_areas, 0, detected_faces, &map_id, &stride,
						&base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

		//access to image
	    vx_uint32 width = 0, height = 0;
	    ERROR_CHECK_STATUS( vxQueryImage( input, VX_IMAGE_ATTRIBUTE_WIDTH,  &width,  sizeof( width ) ) );
	    ERROR_CHECK_STATUS( vxQueryImage( input, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof( height ) ) );
	    vx_rectangle_t rect = { 0, 0, width, height };
	    vx_imagepatch_addressing_t addr_input = { 0 };

		void * ptr_input = NULL;
		ERROR_CHECK_STATUS( vxMapImagePatch( input ,  &rect, 0,&map_id_input  ,&addr_input ,  &ptr_input ,  VX_READ_ONLY  ,NVX_MEMORY_TYPE_CUDA,0 ) );
		cv::gpu::GpuMat mat_input(  height, width, CV_8UC3, ptr_input,  addr_input .stride_y );
		ERROR_CHECK_STATUS( vxUnmapImagePatch( input, map_id_input));

		for (int idx = 0; idx < detected_faces; idx++) {
			vx_rectangle_t face_area = vxArrayItem(vx_rectangle_t, base, idx, stride);
			cv::gpu::GpuMat mat_face = mat_input(cv::Rect(face_area.start_x,face_area.start_y,face_area.end_x,face_area.end_y)) ;
			double blur =  calcBlurriness(mat_face) ;

			cv::gpu::GpuMat img_gpu_ycrcb,resized ;
			cv::gpu::resize(mat_face, resized, cv::Size(224, 224));
			cv::gpu::cvtColor(resized, img_gpu_ycrcb, CV_BGR2YCrCb);
			std::vector<cv::gpu::GpuMat> channels;
			cv::gpu::split(img_gpu_ycrcb, channels);
			cv::gpu::equalizeHist(channels[0], channels[0]);
			merge(channels, img_gpu_ycrcb);
			cv::gpu::GpuMat output;
			cv::gpu::cvtColor(img_gpu_ycrcb, output, CV_YCrCb2BGR);

			const float* features = feature_extractor.GetFeatures(output);
			Prediction p =classifier.Classify(features) ;

			std::cout << "prediction = "<<int(p.first) << ", " << p.second <<", blur = " <<blur<< std::endl;

			user_struct_identication_result identication_res = {
				(vx_uint32)  p.first ,
				(vx_float32) p.second ,
				(vx_float32) blur
			} ;
			ERROR_CHECK_STATUS(vxAddArrayItems(identication_results,1,&identication_res,sizeof(user_struct_identication_result))) ;

		}
		ERROR_CHECK_STATUS(vxUnmapArrayRange(face_areas, map_id));
	}

	return VX_SUCCESS;
}

vx_node userResultValidator(vx_graph  graph,
							 vx_scalar detected_face_number,
							 vx_array  face_areas,
							 vx_array  identication_results,
							 vx_array  validated_results) {

	vx_context context = vxGetContext((vx_reference) graph);
	vx_kernel kernel = vxGetKernelByEnum(context,USER_KERNEL_RESULT_VALIDATOR);
	ERROR_CHECK_OBJECT(kernel);
	vx_node node = vxCreateGenericNode(graph, kernel);
	ERROR_CHECK_OBJECT(node);
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 0, (vx_reference ) detected_face_number));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 1, (vx_reference ) face_areas));
	ERROR_CHECK_STATUS(	vxSetParameterByIndex(node, 2, (vx_reference ) identication_results));
	ERROR_CHECK_STATUS( vxSetParameterByIndex(node, 3, (vx_reference ) validated_results));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return node;
}

//********************************** RESULT VALIDATOR *************************************************************
vx_status VX_CALLBACK result_validator_validator(vx_node node,
		const vx_reference parameters[], vx_uint32 num,
		vx_meta_format metas[]) {

	if (num == 4) {

        // parameters #1 -- scalar of type VX_TYPE_UINT8
        vx_enum type = VX_TYPE_INVALID;
        ERROR_CHECK_STATUS( vxQueryScalar( ( vx_scalar )parameters[0], VX_SCALAR_TYPE, &type, sizeof( type ) ) );

        if( type != VX_TYPE_UINT8 )
        {
            return VX_ERROR_INVALID_TYPE;
        }

        // parameters #2 -- array of type VX_TYPE_RECTANGLE
		ERROR_CHECK_STATUS(	vxQueryArray((vx_array )parameters[1], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
		vx_size num_item ;
        ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[1], VX_ARRAY_NUMITEMS, &num_item, sizeof( num_item ) ) );
		if (type != VX_TYPE_RECTANGLE  ) {
            return VX_ERROR_INVALID_FORMAT;
		}

        // parameters #3 -- array of type VX_TYPE_USER_STRUCT
		ERROR_CHECK_STATUS(	vxQueryArray((vx_array )parameters[2], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[2], VX_ARRAY_NUMITEMS, &num_item, sizeof( num_item ) ) );
		if (type != VX_TYPE_USER_IDENTICATION_RESULT ) {
            return VX_ERROR_INVALID_FORMAT;
		}

		// parameters #4 -- array of type VX_TYPE_USER_STRUCT
		vx_size item_num = MAX_FACE_DETECT;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[3], VX_ARRAY_CAPACITY, &item_num,
						sizeof(item_num)));
		vx_enum object_type = VX_TYPE_USER_VALIDATION_RESULT;
		ERROR_CHECK_STATUS(
				vxSetMetaFormatAttribute(metas[3], VX_ARRAY_ITEMTYPE,
						&object_type, sizeof(object_type)));


	} else {
		// invalid input parameter
		return VX_ERROR_INVALID_PARAMETERS;
	}

	return VX_SUCCESS;
}

vx_status VX_CALLBACK result_validator_init(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	mlp.load(MLP_PATH) ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK result_validator_deinit(vx_node node,
		const vx_reference * parameters, vx_uint32 num) {
	mlp.~CvANN_MLP() ;
	return VX_SUCCESS;
}


vx_status VX_CALLBACK result_validator_process(vx_node node,
		const vx_reference * refs, vx_uint32 num) {

	vx_scalar detected_face_number = (vx_scalar) refs[0];
	vx_array face_areas = (vx_array) refs[1];
	vx_array identication_results = (vx_array) refs[2];
	vx_array validation_results = (vx_array) refs[3];

    vx_uint8  ksize   = 0;
    ERROR_CHECK_STATUS( vxReadScalarValue( detected_face_number, &ksize ) );
	int detected_faces = (int) ksize ;
	if (detected_faces > 0) {
	    //access to identication_results array
		vx_map_id map_id_identication_results;
		vx_size stride_identication = sizeof(user_struct_identication_result);
		void *base_identication = NULL;
		ERROR_CHECK_STATUS(	vxMapArrayRange(identication_results, 0, detected_faces, &map_id_identication_results, &stride_identication,
						&base_identication, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

		//access to detected_faces array
		vx_map_id map_id_detection_results ;
		vx_size stride_detection = sizeof(vx_rectangle_t);
		void *base_detection = NULL;
		ERROR_CHECK_STATUS(	vxMapArrayRange(face_areas, 0, detected_faces, &map_id_detection_results, &stride_detection,
						&base_detection, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

		for (int idx = 0; idx < detected_faces; idx++) {
			vx_rectangle_t face_area = vxArrayItem(vx_rectangle_t, base_detection, idx,	stride_detection);
			user_struct_identication_result face_identication = vxArrayItem(user_struct_identication_result, base_identication, idx, stride_identication);

        	float probibility =  face_identication.class_probability ;
        	float blur = 		 face_identication.face_blurriness ;
        	int dimension     =  (int) face_area.end_x  ;
            cv::Mat response(1, 1, CV_32FC1);
            cv::Mat sample (1, 3, CV_32FC1);
            sample.at<float>  (0) = (probibility  - mins_mlp[0]) / ( maxs_mlp[0] - mins_mlp[0] ) ;
            sample.at<float>  (1) = (blur    - mins_mlp[1]) / ( maxs_mlp[1] - mins_mlp[1] );
            sample.at<float>  (2) = (dimension   - mins_mlp[2]) / ( maxs_mlp[2] - mins_mlp[2] );
//            std::cout <<sample << std::endl ;
            mlp.predict(sample, response);
            float reliable_value = response.at<float>(0,0);
            bool reliable = false ;
//            if(reliable_value >= 0.0 && probibility > 0.9)
//                reliable = true ;
            if(blur < 0.0005 && probibility > 0.45)
            	reliable = true ;

            user_struct_validation_result validation_res = {
            	    face_area.start_x ,
            	    face_area.start_y ,
            	    face_area.end_x,
            	    face_area.end_y,
            	    face_identication.class_label ,
            	    face_identication.class_probability ,
            	    face_identication.face_blurriness ,
            	    (vx_bool) reliable
			} ;
			ERROR_CHECK_STATUS(vxAddArrayItems(validation_results,1,&validation_res,sizeof(user_struct_validation_result))) ;
		}
		ERROR_CHECK_STATUS(vxUnmapArrayRange(face_areas, map_id_detection_results));
		ERROR_CHECK_STATUS(vxUnmapArrayRange(identication_results, map_id_identication_results));

		// make empty the arrays
		ERROR_CHECK_STATUS(vxTruncateArray(face_areas, 0));
		ERROR_CHECK_STATUS(vxTruncateArray(identication_results, 0));
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
