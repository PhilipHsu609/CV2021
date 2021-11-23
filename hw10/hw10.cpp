#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include "Mask.h"

const cv::String lena{"../lena.bmp"};

Mask<int> LOG(int k, double sigma) {
    Mask<int> m(k);
    int offset = k / 2;

    auto g{[sigma](int r, int c) -> double {
        double tmp{(std::pow(r, 2) + std::pow(c, 2)) / (2 * std::pow(sigma, 2))};
        return -(1 - tmp) / (M_PI * std::pow(sigma, 4)) * std::exp(-tmp);
    }};

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double val{g(i - offset, j - offset) * (178 / g(0, 0))};
            m[i].push_back(static_cast<int>(std::round(val - 3e-2)));
        }
    }
    return m;
}

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

    auto x{LOG(11, 1.4)};

    for (auto r : x) {
        for (auto v : r) {
            std::cout << v << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}