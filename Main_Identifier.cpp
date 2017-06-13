#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <fstream>
#include <chrono>
#include <time.h>
#include <algorithm>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>
#include <NVX/nvx_opencv_interop.hpp>

#include "NVXIO/Application.hpp"
#include "NVXIO/ConfigParser.hpp"
#include "NVXIO/FrameSource.hpp"
#include "NVXIO/Render.hpp"
#include "NVXIO/SyncTimer.hpp"
#include "NVXIO/Utility.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>



#include "Graph_User_Nodes_Identifier.h"




vx_status registerUserKernel( vx_context context )
{

    vx_kernel kernel_face_datection = vxAddUserKernel( context,
                                    "app.userkernels.facedetection",
                                    USER_KERNEL_FACE_DETECTION,
									face_detection_process,
                                    3,   // numParams
                                    face_detection_validator,
									face_detection_init,
									face_detection_deinit);

    ERROR_CHECK_OBJECT( kernel_face_datection );
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_datection, 0, VX_INPUT,  VX_TYPE_IMAGE ,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_datection, 1, VX_OUTPUT, VX_TYPE_SCALAR,  VX_PARAMETER_STATE_REQUIRED ) ); // datcted_face_areas number
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_datection, 2, VX_OUTPUT, VX_TYPE_ARRAY ,  VX_PARAMETER_STATE_REQUIRED ) ); // datcted_face_areas
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel_face_datection ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel_face_datection ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.facedetection\n" );

    vx_kernel kernel_face_identication= vxAddUserKernel( context,
                                    "app.userkernels.faceidentication",
                                    USER_KERNEL_FACE_IDENTICATION,
                                    face_identication_process,
                                    4,   // numParams
                                    face_identication_validator,
                                    face_identication_init,
									face_identication_deinit);

    ERROR_CHECK_OBJECT( kernel_face_identication );
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_identication, 0, VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_identication, 1, VX_INPUT, VX_TYPE_SCALAR,  VX_PARAMETER_STATE_REQUIRED ) ); // detected_face number
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_identication, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED ) ); // datcted_face_areas
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_identication, 3, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // identication result
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel_face_identication ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel_face_identication ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.faceidentication\n" );

    vx_kernel kernel_result_validator= vxAddUserKernel( context,
                                    "app.userkernels.resultvalidator",
                                    USER_KERNEL_RESULT_VALIDATOR,
                                    result_validator_process,
                                    4,   // numParams
									result_validator_validator,
									result_validator_init,
									result_validator_deinit);

    ERROR_CHECK_OBJECT( kernel_result_validator );
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_result_validator, 0, VX_INPUT,  VX_TYPE_SCALAR,  VX_PARAMETER_STATE_REQUIRED ) ); // detected_face number
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_result_validator, 1, VX_INPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // datcted_face_areas
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_result_validator, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED ) );  // identication result
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_result_validator, 3, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // validataion result
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel_result_validator ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel_result_validator ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.faceidentication\n" );
    return VX_SUCCESS;
}


//
// Process events
//
struct EventData {
	EventData() :
			stop(false), pause(false) {
	}

	bool stop;
	bool pause;
};

static void keyboardEventCallback(void* eventData, vx_char key, vx_uint32,
		vx_uint32) {
	EventData* data = static_cast<EventData*>(eventData);

	if (key == 27) // escape
			{
		data->stop = true;
	} else if (key == ' ') // space
			{
		data->pause = !data->pause;
	}
}



int main(int argc, char** argv) {

		std::vector<std::string> labels = GetAllLabels() ;
	try {
		nvxio::Application &app = nvxio::Application::get();
//		std::string sourceUri = "rtsp://192.168.20.20/rtpvideo1.sdp";
		std::string sourceUri = SOURCE_INPUT;
//		std::string sourceUri = "/home/pouya/Develop/caffe_old/data/new_data/Camera_Samples/output53.mp4";


		app.init(argc, argv);

		nvxio::ContextGuard context;
		vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
		vxRegisterLogCallback(context, &nvxio::stdoutLogCallback, vx_false_e);

		VX_TYPE_USER_IDENTICATION_RESULT  = vxRegisterUserStruct(context,sizeof(user_struct_identication_result)) ;
		if(VX_TYPE_USER_IDENTICATION_RESULT==VX_TYPE_INVALID) {
			printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", VX_TYPE_INVALID, __LINE__);
			exit(1) ;
		}


		VX_TYPE_USER_VALIDATION_RESULT  = vxRegisterUserStruct(context,sizeof(user_struct_validation_result)) ;
		if(VX_TYPE_USER_IDENTICATION_RESULT==VX_TYPE_INVALID) {
			printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", VX_TYPE_INVALID, __LINE__);
			exit(1) ;
		}

		ERROR_CHECK_STATUS( registerUserKernel( context ) );

		vx_image srcFrame = vxCreateImage(context, ROI_END_X,ROI_END_Y, VX_DF_IMAGE_RGB);

		ERROR_CHECK_OBJECT( srcFrame );

		vx_rectangle_t rect_roi {
			(vx_uint32) ROI_X,
			(vx_uint32) ROI_Y,
			(vx_uint32) ROI_END_X,
			(vx_uint32) ROI_END_Y
		};
		vx_image ROIFrame = vxCreateImageFromROI(srcFrame, &rect_roi);
		ERROR_CHECK_OBJECT( ROIFrame );


	    vx_graph graph      = vxCreateGraph( context );
	    ERROR_CHECK_OBJECT( graph );
	    vx_image yuv_image     = vxCreateVirtualImage( graph, 0, 0, VX_DF_IMAGE_IYUV );
	    vx_image luma_image    = vxCreateVirtualImage( graph, 0, 0, VX_DF_IMAGE_U8 );
	    vx_image resized_image = vxCreateVirtualImage( graph, ROI_END_X/RESIZE_LEVEL_DEPLOY, ROI_END_Y/RESIZE_LEVEL_DEPLOY, VX_DF_IMAGE_U8 );
	    vx_image equlized_image= vxCreateVirtualImage( graph, 0, 0, VX_DF_IMAGE_U8 );

	    vx_uint8 init_face_number = 0 ;
	    vx_scalar detected_face_number = vxCreateScalar(context,VX_TYPE_UINT8,&init_face_number) ;

	    vx_array face_areas = vxCreateVirtualArray(graph ,VX_TYPE_RECTANGLE , MAX_FACE_DETECT) ;
		vx_array identication_results = vxCreateVirtualArray(graph ,VX_TYPE_USER_IDENTICATION_RESULT , MAX_FACE_DETECT) ;
		vx_array validation_results = vxCreateArray(context ,VX_TYPE_USER_VALIDATION_RESULT , MAX_FACE_DETECT) ;

	    ERROR_CHECK_OBJECT( yuv_image );
	    ERROR_CHECK_OBJECT( luma_image );
	    ERROR_CHECK_OBJECT( resized_image );
	    ERROR_CHECK_OBJECT( equlized_image );

	    ERROR_CHECK_OBJECT( detected_face_number );

	    ERROR_CHECK_OBJECT( face_areas );
	    ERROR_CHECK_OBJECT( identication_results );
	    ERROR_CHECK_OBJECT( validation_results );

	      vx_node nodes[] =
	      {
	          vxColorConvertNode(   graph, ROIFrame, luma_image ),
	          vxScaleImageNode( graph, luma_image,resized_image, VX_INTERPOLATION_AREA ),
			  vxEqualizeHistNode(graph,resized_image,equlized_image) ,
	  		  userFaceDetection(   graph, equlized_image,detected_face_number ,face_areas ),
	  		  userFaceIdentication(graph ,ROIFrame,detected_face_number,face_areas,identication_results),
			  userResultValidator(graph,detected_face_number,face_areas,identication_results,validation_results)
	      };
	      for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
	      {
	          ERROR_CHECK_OBJECT( nodes[i] );
	          ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
	      }
	      ERROR_CHECK_STATUS( vxReleaseImage(  &yuv_image ) );
	      ERROR_CHECK_STATUS( vxReleaseImage(  &luma_image ) );
	      ERROR_CHECK_STATUS( vxReleaseImage(  &resized_image ) );
	      ERROR_CHECK_STATUS( vxReleaseImage(  &equlized_image ) );
	      ERROR_CHECK_STATUS( vxReleaseArray(  &face_areas ) );
	      ERROR_CHECK_STATUS( vxReleaseArray(  &identication_results ) );


	      ERROR_CHECK_STATUS( vxVerifyGraph(   graph ) );

			std::unique_ptr<nvxio::FrameSource> frameSource(
					nvxio::createDefaultFrameSource(context, sourceUri));

			if (!frameSource || !frameSource->open()) {
				std::cerr << "Error: cannot open frame source!" << std::endl;
				return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
			}

			if (frameSource->getSourceType()
					== nvxio::FrameSource::SINGLE_IMAGE_SOURCE) {
				std::cerr << "Can't work on a single image." << std::endl;
				return nvxio::Application::APP_EXIT_CODE_INVALID_FORMAT;
			}

			nvxio::FrameSource::Parameters frameConfig =
					frameSource->getConfiguration();

			//
			// Create a Render
			//

			std::unique_ptr<nvxio::Render> render = nvxio::createDefaultRender(
					context, "Face Detection and Identication",
					frameConfig.frameWidth, frameConfig.frameHeight);

			if (!render) {
				std::cerr << "Error: Cannot create render!" << std::endl;
				return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
			}

			EventData eventData;
			render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);


		//
		// Create algorithm
		//

		nvxio::FrameSource::FrameStatus frameStatus;
		do {
			frameStatus = frameSource->fetch(srcFrame);
		} while (frameStatus == nvxio::FrameSource::TIMEOUT);
		if (frameStatus == nvxio::FrameSource::CLOSED) {
			std::cerr << "Source has no frames" << std::endl;
			return nvxio::Application::APP_EXIT_CODE_NO_FRAMESOURCE;
		}

		//
		// Main loop
		//

		std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
		syncTimer->arm(1. / app.getFPSLimit());


		double time_total,time_graph ;
		nvx::Timer timer_graph,timer_total;

//		std::vector<vx_rectangle_t> face_rects ;

		while ( !eventData.stop ) {
			if (!eventData.pause) {
				timer_total.tic() ;

				frameStatus = frameSource->fetch(srcFrame);

				timer_graph.tic() ;
				ERROR_CHECK_STATUS( vxProcessGraph( graph ) );
				time_graph = timer_graph.toc() ;

			    vx_uint8  ksize   = 0;
			    ERROR_CHECK_STATUS( vxReadScalarValue( detected_face_number, &ksize ) );
			    int face_number = (int )ksize ;

				syncTimer->synchronize();

				render->putImage(ROIFrame);

		        if(face_number > 0) {
			        void* ptr_validation_result ;
			        vx_size stride_validation_result = sizeof(user_struct_validation_result);
			        vx_map_id map_id_validation_result ;
			        ERROR_CHECK_STATUS(vxMapArrayRange(validation_results,0,face_number,&map_id_validation_result,&stride_validation_result,&ptr_validation_result,VX_READ_ONLY,VX_MEMORY_TYPE_HOST,0)) ;
			        for (int idx=0 ; idx < face_number ; ++idx) {
			        	user_struct_validation_result validation_result = vxArrayItem(user_struct_validation_result, ptr_validation_result, idx,	stride_validation_result);
		                vx_rectangle_t face_rect = { validation_result.start_x, validation_result.start_y,  validation_result.start_x +validation_result.end_x,validation_result.start_y+ validation_result.end_y };

		                if (((bool) validation_result.validated)) {
		                	nvxio::Render::DetectedObjectStyle style = {labels[(int)validation_result.class_label],{0,255,0,255} , 2, false };
		                	render->putObjectLocation(face_rect, style);
		                }
		                else {
		                	nvxio::Render::DetectedObjectStyle style = {"unknown" ,{255,0,0,255} , 2, false };
		                	render->putObjectLocation(face_rect, style);
		                }


			        }

			        ERROR_CHECK_STATUS(vxUnmapArrayRange(validation_results,map_id_validation_result)) ;
			        ERROR_CHECK_STATUS(vxTruncateArray(validation_results, 0));

		        }

		        time_total = timer_total.toc();

			}
			else {
				render->putImage(ROIFrame);
			}



			//
			// Show performance statistics
			//

			nvxio::Render::MotionFieldStyle mfStyle = { { 0u, 255u, 255u, 255u } // color
			};

			std::ostringstream msg;
			msg << std::fixed << std::setprecision(1);

			msg << "Resolution: " << frameConfig.frameWidth << 'x'
					<< frameConfig.frameHeight << std::endl;
			msg << "Graph Time: " << time_graph << " ms / "
					<< 1000.0 / time_graph << " FPS" << std::endl;
			msg << "Total Time: " << time_total << " ms / " << 1000.0 / time_total
					<< " FPS" << std::endl;
			msg << "Space - pause/resume" << std::endl;
			msg << "Esc - close the sample";

			nvxio::Render::TextBoxStyle textStyle = {
					{ 255u, 255u, 255u, 255u }, // color
					{ 0u, 0u, 0u, 127u }, // bgcolor
					{ 10u, 10u } // origin
			};

			render->putTextViewport(msg.str(), textStyle);

			if (!render->flush()) {
				eventData.stop = true;
			}

		}



		//
		// Release all objects
		//
		ERROR_CHECK_STATUS( vxReleaseImage(&srcFrame));
		ERROR_CHECK_STATUS( vxReleaseArray(  &validation_results ) );
	    ERROR_CHECK_STATUS( vxReleaseScalar( &detected_face_number ) );

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return nvxio::Application::APP_EXIT_CODE_ERROR;
	}

	return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}



