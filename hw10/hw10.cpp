#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include "Mask.h"

const cv::String lena{"../lena.bmp"};

template <class T>
std::vector<std::vector<int>> applyMask(const cv::Mat &image, const Mask<T> &mask, int offset) {
    int m = image.rows, n = image.cols;
    std::vector<std::vector<int>> output(m, std::vector<int>(n, 0));
    cv::Mat pad;
    cv::copyMakeBorder(image, pad, offset, offset, offset, offset, cv::BORDER_REPLICATE);

    for (int i = offset; i < m + offset; i++) {
        for (int j = offset; j < n + offset; j++) {
            T conv{};
            for (int k = 0; k < mask.size(); k++) {
                for (int l = 0; l < mask.size(); l++) {
                    conv += static_cast<T>(pad.at<uchar>(i + k - offset, j + l - offset)) * mask[k][l];
                }
            }
            output[i - offset][j - offset] = conv;
        }
    }

    return output;
}

cv::Mat zeroCorssing(const std::vector<std::vector<int>> &image, int threshold) {
    int m = image.size(), n = image[0].size();
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    const std::vector<std::vector<int>> dirs{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int i = 0; i < m; i++) {
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            uchar p = 255;
            if (image[i][j] >= threshold) {
                for (auto &dir : dirs) {
                    int ii{i + dir[0]}, jj{j + dir[1]};
                    if (ii >= 0 && jj >= 0 && ii < m && jj < n)
                        if (image[ii][jj] <= -threshold) {
                            p = 0;
                            break;
                        }
                }
            }
            dst[j] = p;
        }
    }

    return image_;
}

int main() {
    auto image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};

    cv::Mat M;

    M = zeroCorssing(applyMask(image, L4, 1), 15);
    cv::imwrite("L4.bmp", M);

    M = zeroCorssing(applyMask(image, L8, 1), 15);
    cv::imwrite("L8.bmp", M);

    M = zeroCorssing(applyMask(image, mvL, 1), 20);
    cv::imwrite("mvL.bmp", M);

    M = zeroCorssing(applyMask(image, LOG, 5), 3000);
    cv::imwrite("LOG.bmp", M);

    M = zeroCorssing(applyMask(image, DOG, 5), 1);
    cv::imwrite("DOG.bmp", M);

    return 0;
}