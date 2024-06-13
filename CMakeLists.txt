cmake_minimum_required(VERSION 3.10)

project(AudioTflite)

# Set C++ standard to C++11
# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(/home/mcw/ajith/audio_source_sep/CPP_inference/001_eigen_lib/eigen)
# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
# /home/mcw/ajith/opencv_cuda/opencvinstall
# Include the directory for header files
include_directories("/home/mcw/ajith/opencv_cuda/opencvinstall")
include_directories("/home/mcw/ajith/opencv_cuda/opencv_contrib")
# Add the directory for library files
link_directories("/home/mcw/ajith/opencv_cuda/opencvinstall/lib")

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)

set(OPENCV_LIBS
    opencv_cudabgsegm
    opencv_cudaobjdetect
    opencv_cudastereo
    opencv_stitching
    opencv_cudafeatures2d
    opencv_superres
    opencv_videostab
    opencv_cudaoptflow
    opencv_cudalegacy
    opencv_cudawarping
    opencv_aruco
    opencv_bgsegm
    opencv_bioinspired
    opencv_ccalib
    opencv_dnn_objdetect
    opencv_dpm
    opencv_highgui
    opencv_videoio
    opencv_face
    opencv_freetype
    opencv_fuzzy
    opencv_hdf
    opencv_hfs
    opencv_img_hash
    opencv_line_descriptor
    opencv_optflow
    opencv_reg
    opencv_rgbd
    opencv_saliency
    opencv_sfm
    opencv_stereo
    opencv_structured_light
    opencv_phase_unwrapping
    opencv_surface_matching
    opencv_tracking
    opencv_datasets
    opencv_text
    opencv_dnn
    opencv_plot
    opencv_xfeatures2d
    opencv_shape
    opencv_video
    opencv_ml
    opencv_ximgproc
    opencv_xobjdetect
    opencv_objdetect
    opencv_calib3d
    opencv_imgcodecs
    opencv_features2d
    opencv_flann
    opencv_xphoto
    opencv_photo
    opencv_cudaimgproc
    opencv_cudafilters
    opencv_imgproc
    opencv_cudaarithm
    opencv_core
    opencv_cudev
    opencv_dnn
    opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_video
    opencv_imgcodecs
    opencv_videoio
    opencv_highgui
)

# Define the source files for the executable
set(SOURCES
    main.cpp
)

# Add an executable target
add_executable(TfliteAd ${SOURCES})

# Optionally, you can specify additional include directories
# include_directories(path/to/include)

# Optionally, you can link to external libraries
target_link_libraries(TfliteAd tensorflowlite ${OPENCV_LIBS})

# Specify the output directory for the executable (optional)
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Specify the output name of the executable (optional)
# set_target_properties(myapp PROPERTIES OUTPUT_NAME myapp)