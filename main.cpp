#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;

struct Box {
    float x;  // x-coordinate of the center
    float y;  // y-coordinate of the center
    float w;  // width of the box
    float h;  // height of the box
};

float IoU(const cv::Rect& box1, const cv::Rect& box2) {
    float x1_min = box1.x - box1.width / 2;
    float y1_min = box1.y - box1.height / 2;
    float x1_max = box1.x + box1.width / 2;
    float y1_max = box1.y + box1.height / 2;

    float x2_min = box2.x - box2.width / 2;
    float y2_min = box2.y - box2.height / 2;
    float x2_max = box2.x + box2.width / 2;
    float y2_max = box2.height + box2.height / 2;

    float inter_x_min = max(x1_min, x2_min);
    float inter_y_min = max(y1_min, y2_min);
    float inter_x_max = min(x1_max, x2_max);
    float inter_y_max = min(y1_max, y2_max);

    float inter_area = max(0.0f, inter_x_max - inter_x_min) * max(0.0f, inter_y_max - inter_y_min);
    float box1_area = (x1_max - x1_min) * (y1_max - y1_min);
    float box2_area = (x2_max - x2_min) * (y2_max - y2_min);

    return inter_area / (box1_area + box2_area - inter_area);
}

vector<int> non_max_suppression(const vector<cv::Rect>& boxes, const vector<float>& scores, float iou_threshold) {
    vector<int> indices;
    vector<int> sorted_indices(scores.size());

    // Manually filling sorted_indices with indices from 0 to scores.size() - 1
    for (size_t i = 0; i < scores.size(); ++i) {
        sorted_indices[i] = i;
    }

    // Sorting indices based on corresponding scores in descending order
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j) {
        return scores[i] > scores[j];
    });

    vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        int idx = sorted_indices[i];
        if (suppressed[idx]) continue;  // Skip already suppressed boxes

        indices.push_back(idx);  // Add index of box to keep

        for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
            int idx_j = sorted_indices[j];
            if (suppressed[idx_j]) continue;  // Skip if already suppressed

            // Suppress the box if IoU is above the threshold
            if (IoU(boxes[idx], boxes[idx_j]) > iou_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }

    return indices;
}




int main() {
    // Load the image
    cv::Mat image = cv::imread("../test_image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    std::vector<string> coco_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    };
    // Resize the image
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640), cv::INTER_AREA);

    // Load the TensorFlow Lite model.
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("./yolov8n_saved_model/yolov8n_float32.tflite");
    if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    // Build the TensorFlow Lite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create Interpreter." << std::endl;
        return 1;
    }

    // Allocate tensor buffers
    interpreter->AllocateTensors();

    int input_tensor_index = interpreter->inputs()[0];
    int output_tensor_index = interpreter->outputs()[0];

    // Model input size is 320x320x3 (width x height x channels)
    int input_width = 640;
    int input_height = 640;
    int input_channels = 3;
    int input_size = input_width * input_height * input_channels;

    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);

    // Get the dimensions of the input tensor
    int input_dims = input_tensor->dims->size;

    // Double check the input tensor size is correct (1, 640, 640, 3)
    int batch_size = input_tensor->dims->data[0];
    int height = input_tensor->dims->data[1];
    int width = input_tensor->dims->data[2];
    int channels = input_tensor->dims->data[3];

    // Preprocess the input image
    cv::Mat input_float;
    resized_image.convertTo(input_float, CV_32FC3, 1.0f / 255.0f); // Convert to float and normalize

    // Copy input image data to input tensor
    float* input_data = interpreter->typed_input_tensor<float>(0);
    memcpy(input_data, input_float.data, input_size * sizeof(float));


    // Run inference
    interpreter->Invoke();

    TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
    // Post-Processing - extracting the results
    float* output_data = interpreter->typed_output_tensor<float>(0);

    for (int i = 0; i < output_box->dims->size; ++i)
    {
        cout << "DIM IS " << output_box->dims->data[i] << endl;
    }

    // Get output dimensions and reshape for postprocessing
    int _out = interpreter->outputs()[0];
    TfLiteIntArray *out_dims = interpreter->tensor(_out)->dims;
    int out_row = out_dims->data[1];
    int out_colum = out_dims->data[2];

    float reshaped[out_colum][out_row];
    for (int i = 0; i < out_row; i++) {
        for (int j = 0; j < out_colum; j++) {
            reshaped[j][i] = output_data[i * out_colum + j];
        }
    }
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat reshaped_yolo(out_colum, out_row, CV_32F, reshaped);
    float *data = (float *)reshaped_yolo.data;
    float confThreshold = 0.2;

    float w_scale = 1;
    int rows = out_colum;
    int cols = out_row;
    int resizeScales = 640;
    for (int i = 0; i < rows; i++)
    {
        std::vector<float> it(data + i * cols, data + (i + 1) * cols);
        float confidence;
        int classId;

        int bestClassId = 4;
        float bestConf = 0;

        for (int i = 4; i < 80 + 4; i++)
        {
            if (it[i] > bestConf)
            {
                bestConf = it[i];
                bestClassId = i - 4;
            }
        }

        if (bestConf > confThreshold)
        {
            float x = (float)(it[0]);
            float y = (float)(it[1]);
            float w = (float)(it[2]);
            float h = (float)(it[3]);

            int left = int((x - 0.5 * w) * resizeScales);
            int top = int((y - 0.5 * h) * resizeScales);

            int width = int(w * resizeScales);
            int height = int(h * resizeScales);
            std::cout << it[0] << " "<<it[1] <<" "<< it[2]<< " "<< it[3]<< std::endl;
            boxes.emplace_back(left, top, width, height);
            confidences.emplace_back(bestConf);
            class_ids.emplace_back(bestClassId);
        }
    }


    std::vector<int> nms_result;

    for (size_t j = 0; j < boxes.size(); ++j) {
        if (confidences[j] > 0.5) {
            std::cout << "Box " << j << ": " << boxes[j] << ", Class: " << class_ids[j] << ", Confidence: " << confidences[j] << std::endl;
        }
    }


    // // Apply non-maximum suppression to filter overlapping boxes
    float iou_threshold = 0.5;
    vector<int> nms_indices = non_max_suppression(boxes, confidences, iou_threshold);

    // // Draw the boxes on the original image
    for (int i : nms_indices) {
        cv::rectangle(resized_image, boxes[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(resized_image, coco_names[class_ids[i]], cv::Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::putText(resized_image, to_string(confidences[i]), cv::Point(boxes[i].x, boxes[i].y + 20), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    // // Save or display the result image
    cv::imwrite("../output_image.jpg", resized_image);

    return 0;
}
