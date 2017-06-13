echo "Compiling Face Identifier"
g++ `pkg-config --cflags opencv ` -std=c++11 -I./ Classifier.cpp FeatureExtractorDeploy.cpp -O3 Main_Identifier.cpp -o Identifier_IP `pkg-config visionworks visionworks-tracking  opencv nvxio --libs` -lprotobuf -lpthread -lcaffe -lboost_system -lglog `python-config --ldflags` -ljpeg -lpng -lopenblas -ldlib -lpng12  -llapack -lboost_filesystem
echo "Compiling Face Tracker"
g++ `pkg-config --cflags opencv ` -std=c++11 -I./ Classifier.cpp FeatureExtractorDeploy.cpp -O3 Main_Tracker.cpp -o Tracker_IP `pkg-config visionworks visionworks-tracking  opencv nvxio --libs` -lprotobuf -lpthread -lcaffe -lboost_system -lglog `python-config --ldflags` -ljpeg -lpng -lopenblas -ldlib -lpng12  -llapack -lboost_filesystem

