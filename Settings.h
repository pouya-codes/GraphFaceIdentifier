/*
 * Settings.h
 *
 *  Created on: Feb 2, 2017
 *      Author: pouya
 */

#ifndef SRC_SETTINGS_H_
#define SRC_SETTINGS_H_


#define MAX_FACE_DETECT 10
#define RESIZE_LEVEL_DEPLOY 3
#define IMAGE_PAD_PERSENT_DEPLOY 0.0

//#define CASCADE_FILE_PATH "/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/haarcascade_frontalface_alt2.xml"
//#define CASCADE_FILE_PATH "/home/pouya/Develop/opencv-2.4.13/data/haarcascades_GPU/haarcascade_frontalface_alt.xml"
//#define CASCADE_FILE_PATH "/home/pouya/Develop/opencv-2.4.13/data/haarcascades_GPU/haarcascade_frontalface_alt_tree.xml"
#define CASCADE_FILE_PATH "/home/pouya/Develop/opencv-2.4.13/data/haarcascades_GPU/haarcascade_frontalface_default.xml"
//build-FaceIdentifierResNet-Desktop-Release

#define NET_PATH_CLASSIFIER "/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/classifier/deploy.prototxt"
#define NET_WEIGHTS_CLASSIFIER	"/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/classifier/classifier_layer_iter_100.caffemodel"


//#define NET_PATH_CLASSIFIER "/home/pouya/Develop/qt/build-FaceIdentifierResNet-Desktop-Release/model/classifier/deploy.prototxt"
//#define NET_WEIGHTS_CLASSIFIER	"/home/pouya/Develop/qt/build-FaceIdentifierResNet-Desktop-Release/model/classifier/classifier_layer_iter_100.caffemodel"


//#define NET_PATH_VGG "/home/pouya/Develop/qt/build-FaceIdentifierResNet-Desktop-Release/model/res/ResNet-101-deploy_augmentation.prototxt"
//#define NET_WEIGHTS_VGG	"/home/pouya/Develop/qt/build-FaceIdentifierResNet-Desktop-Release/model/res/snap_resnet__iter_120000.caffemodel"

#define NET_PATH_VGG "/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/vgg/deploy.prototxt"
#define NET_WEIGHTS_VGG	"/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/vgg/VGG_FACE.caffemodel"

#define MLP_PATH "/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/model/mlp_weights.xmls"

//#define SOURCE_INPUT "/home/pouya/Develop/caffe_old/data/new_data/Camera_Samples/output.mp4"
#define SOURCE_INPUT "device:///v4l2?index=0"
//#define SOURCE_INPUT "rtsp://192.168.20.20/rtpvideo1.sdp"
//#define SOURCE_INPUT "/home/pouya/MOV_20170122_133824_135.mp4"


#define ROI_X 0
#define ROI_Y 0
#define ROI_END_X 1280
#define ROI_END_Y 720



static int GetLastID() {

    if(!boost::filesystem::exists("/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/databases")) return 0 ;
    std::string db_name ="/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/databases/labels_db" ;
	if(!boost::filesystem::exists(db_name)) return 0;

	boost::shared_ptr<db::DB> db(db::GetDB("lmdb"));

	int counter = 0 ;
	db->Open(db_name, db::READ) ;
	boost::shared_ptr<db::Cursor> cursor_read(db->NewCursor());

	while(cursor_read->valid()) {
//		std::cout << cursor_read->key() << " - "<<cursor_read->value() << std::endl ;
		cursor_read->Next() ;
		counter++ ;
	}
	cursor_read->~Cursor() ;
	db->Close();
	return counter ;
}

static std::vector<std::string> GetAllLabels() {

	std::vector<std::string> Labels ;
    if(!boost::filesystem::exists("/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/databases")) return Labels ;
    std::string db_name ="/home/pouya/Develop/qt/build-FaceIdentifier-Desktop-Release/databases/labels_db" ;
	if(!boost::filesystem::exists(db_name)) return Labels;
	int last_id = GetLastID() ;
	std::string myArray[last_id];

	boost::shared_ptr<db::DB> db(db::GetDB("lmdb"));


	db->Open(db_name, db::READ) ;
	boost::shared_ptr<db::Cursor> cursor_read(db->NewCursor());

	while(cursor_read->valid()) {
		myArray[std::stoi(cursor_read->key())] = cursor_read->value() ;
		cursor_read->Next() ;

	}
	cursor_read->~Cursor() ;
	db->Close();
	for (int idx = 0; idx < last_id; ++idx) {
		Labels.push_back(myArray[idx]) ;
	}
	return Labels ;

}




#endif /* SRC_SETTINGS_H_ */
