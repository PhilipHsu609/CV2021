#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include "Mask.h"

const cv::String lena{"../lena.bmp"};

template <class T>
cv::Mat edgeDetect(const cv::Mat &image, const std::vector<Mask<T>> &masks, int threshold, int offset = 0) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));
    cv::Mat pad;
    cv::copyMakeBorder(image, pad, offset, offset, offset, offset, cv::BORDER_REPLICATE);

    for (int i = offset; i < m + offset; i++) {
        uchar *dst{image_.ptr<uchar>(i - offset)};
        for (int j = offset; j < n + offset; j++) {
            T acc{}, max{};
            for (const auto &mask : masks) {
                T conv{};
                for (int k = 0; k < mask.size(); k++) {
                    for (int l = 0; l < mask.size(); l++) {
                        conv += pad.at<uchar>(i + k - offset, j + l - offset) * mask[k][l];
                    }
                }
                acc += conv * conv;
                max = std::max<T>(max, conv);
            }
            T grad = (masks.size() != 2) ? max : std::sqrt(acc);
            dst[j - offset] = (grad >= threshold) ? 0 : 255;
        }
    }

    return image_;
}

int main() {
    auto image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};

    cv::imwrite("robert.bmp", edgeDetect(image, Detector::Robert, 30));
    cv::imwrite("prewitt.bmp", edgeDetect(image, Detector::Prewitt, 90, 1));
    cv::imwrite("sobel.bmp", edgeDetect(image, Detector::Sobel, 120, 1));
    cv::imwrite("frei.bmp", edgeDetect(image, Detector::FreiAndChen, 100, 1));
    cv::imwrite("kirsch.bmp", edgeDetect(image, Detector::Kirsch, 400, 1));
    cv::imwrite("robinson.bmp", edgeDetect(image, Detector::Robinson, 120, 1));
    cv::imwrite("babu.bmp", edgeDetect(image, Detector::NevatiaAndBabu, 22222, 2));

    return 0;
}