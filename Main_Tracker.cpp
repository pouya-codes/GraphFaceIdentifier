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

#include "Graph_User_Nodes_Tracker.h"




vx_status registerUserKernel( vx_context context )
{

    vx_kernel kernel_face_tracker = vxAddUserKernel( context,
                                    "app.userkernels.facetracker",
                                    USER_KERNEL_FACE_TRACKER,
									face_tracker_process,
                                    3,   // numParams
									face_tracker_validator,
									face_tracker_init,
									face_tracker_deinit);

    ERROR_CHECK_OBJECT( kernel_face_tracker );
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_tracker, 0, VX_INPUT,  VX_TYPE_IMAGE ,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_tracker, 1, VX_INPUT,  VX_TYPE_IMAGE ,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel_face_tracker, 2, VX_BIDIRECTIONAL,  VX_TYPE_ARRAY ,  VX_PARAMETER_STATE_REQUIRED ) ); // tracker_array
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel_face_tracker ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel_face_tracker ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.facetracker\n" );

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
//		std::string sourceUri = "device:///v4l2?index=0";
		std::string sourceUri = SOURCE_INPUT;


		app.init(argc, argv);

		nvxio::ContextGuard context;
		vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
		vxRegisterLogCallback(context, &nvxio::stdoutLogCallback, vx_false_e);

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


		VX_TYPE_USER_EVENT_TRACKER  = vxRegisterUserStruct(context,sizeof(user_struct_event_tracker)) ;
		if(VX_TYPE_USER_EVENT_TRACKER==VX_TYPE_INVALID) {
			printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", VX_TYPE_INVALID, __LINE__);
			exit(1) ;
		}

		// add tracker
		_user_struct_event_tracker faceTracker;
		faceTracker.frameWidth = frameConfig.frameWidth;
		faceTracker.frameHeight = frameConfig.frameHeight;
		faceTracker.frame = vxCreateImage(context, frameConfig.frameWidth,
				frameConfig.frameHeight, frameConfig.format);

		KeypointObjectTrackerParams params;
		std::string msg;
		std::string configFile = "/home/pouya/Develop/workspace/Graph_Based/src/tracking_config.ini";
		if (!read_tracker_config_file(configFile, params, msg)) {
			std::cout << msg << std::endl;
			return nvxio::Application::APP_EXIT_CODE_INVALID_VALUE;
		}


		std::unique_ptr<ObjectTrackerWithFeaturesInfo> tracker(
				nvxTrackingCreateKeypointObjectTrackerWithFeaturesInfo(context,
						params));

		if (!tracker) {
			std::cerr << "Error: Can't initialize object tracker algorithm."
					<< std::endl;
			return nvxio::Application::APP_EXIT_CODE_CAN_NOT_CREATE;
		}

		faceTracker.tracker = tracker.get();

		vx_array tracker_array = vxCreateArray(context,VX_TYPE_USER_EVENT_TRACKER,1) ;
	    ERROR_CHECK_OBJECT( tracker_array) ;
	    ERROR_CHECK_STATUS(vxAddArrayItems(tracker_array,1,&faceTracker,sizeof(user_struct_event_tracker))) ;


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



	    ERROR_CHECK_OBJECT( yuv_image );
	    ERROR_CHECK_OBJECT( luma_image );
	    ERROR_CHECK_OBJECT( resized_image );
	    ERROR_CHECK_OBJECT( equlized_image );


	      vx_node nodes[] =
	      {
	          vxColorConvertNode(   graph, ROIFrame, luma_image ),
	          vxScaleImageNode( graph, luma_image,resized_image, VX_INTERPOLATION_AREA ),
			  vxEqualizeHistNode(graph,resized_image,equlized_image) ,
	  		  userFaceTracker(   graph, equlized_image,ROIFrame ,tracker_array ),
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


	      ERROR_CHECK_STATUS( vxVerifyGraph(   graph ) );

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

		nvx::Timer timer_total,timer_graph_tracker,timer_graph_identifier;
		double time_total ,time_graph_tracker,time_graph_identifier;
		_user_struct_event_tracker face_tracker;

		while ( !eventData.stop ) {
			if (!eventData.pause) {
				timer_total.tic() ;

				frameStatus = frameSource->fetch(srcFrame);


		        void* ptr_tracker ;
		        vx_size stride_tracker = sizeof(_user_struct_event_tracker);
		        vx_map_id map_id_tracker ;
		        ERROR_CHECK_STATUS(vxMapArrayRange(tracker_array,0,1,&map_id_tracker,&stride_tracker,&ptr_tracker,VX_READ_AND_WRITE,VX_MEMORY_TYPE_HOST,0)) ;

		        face_tracker = vxArrayItem(_user_struct_event_tracker, ptr_tracker, 0,	stride_tracker);
		        face_tracker.frame = srcFrame;


				if (face_tracker.objects.size()>0)
				{
					timer_graph_tracker.tic() ;
					face_tracker.tracker->process(face_tracker.frame);
					time_graph_tracker= timer_graph_tracker.toc() ;
				}

				auto colorIt = face_tracker.colors.begin();
				auto lableIt = face_tracker.labels.begin();
				for (auto it = face_tracker.objects.begin();
						it != face_tracker.objects.end();) {
					ObjectTrackerWithFeaturesInfo::TrackedObject* obj = *it;
					ObjectTracker::ObjectStatus status = obj->getStatus();
					vx_rectangle_t rect = obj->getLocation();
					unsigned int area = (rect.end_x - rect.start_x)
							* (rect.end_y - rect.start_y);
					unsigned int frameArea = face_tracker.frameWidth
							* face_tracker.frameHeight;

					if (status != ObjectTracker::LOST && area < frameArea / 12) {
						++it;
						++colorIt;
						++lableIt;
					} else {
						it = face_tracker.objects.erase(it);
						colorIt = face_tracker.colors.erase(colorIt);
						lableIt = face_tracker.labels.erase(lableIt);
						face_tracker.tracker->removeObject(obj);
					}
				}
				vxArrayItem(_user_struct_event_tracker, ptr_tracker, 0,	stride_tracker) = face_tracker;
		        ERROR_CHECK_STATUS(vxUnmapArrayRange(tracker_array,map_id_tracker)) ;
		        timer_graph_identifier.tic();

				ERROR_CHECK_STATUS( vxProcessGraph( graph ) );

				time_graph_identifier= timer_graph_identifier.toc() ;


//		        if(face_number > 0) {
//			        void* ptr_validation_result ;
//			        vx_size stride_validation_result = sizeof(user_struct_validation_result);
//			        vx_map_id map_id_validation_result ;
//			        ERROR_CHECK_STATUS(vxMapArrayRange(validation_results,0,face_number,&map_id_validation_result,&stride_validation_result,&ptr_validation_result,VX_READ_ONLY,VX_MEMORY_TYPE_HOST,0)) ;
//			        for (int idx=0 ; idx < face_number ; ++idx) {
//			        	user_struct_validation_result validation_result = vxArrayItem(user_struct_validation_result, ptr_validation_result, idx,	stride_validation_result);
//		                std::cout  << validation_result.class_label <<" -- " <<validation_result.class_probability << " -- " <<(int)validation_result.end_x<<" -- "<<validation_result.validated <<std::endl ;
//			        }
//
//			        ERROR_CHECK_STATUS(vxUnmapArrayRange(validation_results,map_id_validation_result)) ;
//			        ERROR_CHECK_STATUS(vxTruncateArray(validation_results, 0));
//
//
//			        if(face_number==1) {
//			        	total_time_ms+=total_time.toc() ;
//			        	frame_sample++ ;
////				        std::cout <<frame_sample<<'-'  <<total_time_ms << std::endl ;
//			        }
//		        }

				render->putImage(ROIFrame);

		        ERROR_CHECK_STATUS(vxMapArrayRange(tracker_array,0,1,&map_id_tracker,&stride_tracker,&ptr_tracker,VX_READ_ONLY,VX_MEMORY_TYPE_HOST,0)) ;

		        face_tracker = vxArrayItem(_user_struct_event_tracker, ptr_tracker, 0,	stride_tracker);



				for (size_t i = 0; i < face_tracker.objects.size(); i++) {
					vx_rectangle_t rect =
							face_tracker.objects[i]->getLocation();
					const Scalar &c = face_tracker.colors[i];
					nvxio::Render::DetectedObjectStyle style = {
							face_tracker.labels.at(i)<=-1 ? face_tracker.labels.at(i)==-1 ? "unknown" : "blur" : labels[face_tracker.labels.at(i)],
									{ c.values[0],
							c.values[1], c.values[2], 255 }, 2, false };
					render->putObjectLocation(rect, style);

//					std::cout << face_tracker.labels.at(i)<<std::endl ;

				}
				ERROR_CHECK_STATUS(vxUnmapArrayRange(tracker_array,map_id_tracker)) ;


				time_total = timer_total.toc();

			}
			else {
				render->putImage(ROIFrame);






				for (size_t i = 0; i < face_tracker.objects.size(); i++) {
					vx_rectangle_t rect =
							face_tracker.objects[i]->getLocation();
					const Scalar &c = face_tracker.colors[i];
					nvxio::Render::DetectedObjectStyle style = {
							face_tracker.labels.at(i)<=-1 ? face_tracker.labels.at(i)==-1 ? "unknown" : "blur" : labels[face_tracker.labels.at(i)],
									{ c.values[0],
							c.values[1], c.values[2], 255 }, 2, false };
					render->putObjectLocation(rect, style);


				}

			}



//			std::cout << "Display Time : " << total_ms << " ms" << std::endl
//					<< std::endl;

			syncTimer->synchronize();


			//
			// Show performance statistics
			//

			nvxio::Render::MotionFieldStyle mfStyle = { { 0u, 255u, 255u, 255u } // color
			};

			std::ostringstream msg;
			msg << std::fixed << std::setprecision(1);

			msg << "Resolution: " << frameConfig.frameWidth << 'x'
					<< frameConfig.frameHeight << std::endl;
			msg << "Tracker Graph Time: " << time_graph_tracker << " ms / "
					<< 1000.0 / time_graph_tracker << " FPS" << std::endl;
			msg << "Identifier Graph Time: " << time_graph_identifier << " ms / "
					<< 1000.0 / time_graph_identifier << " FPS" << std::endl;
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


	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return nvxio::Application::APP_EXIT_CODE_ERROR;
	}

	return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}


