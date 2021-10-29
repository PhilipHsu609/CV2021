#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using Kernel = std::vector<std::vector<int>>;

const cv::String lena{"../lena.bmp"};

const Kernel octagonKernel() {
    Kernel kernel;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (i * j != 4 && i * j != -4)
                kernel.push_back({i, j});
        }
    }
    return kernel;
}

cv::Mat dilation(const cv::Mat &image, const Kernel &k) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            uchar tmp = 0;
            for (auto &p : k) {
                int y{i + p[0]}, x{j + p[1]};
                if (y >= 0 && x >= 0 && y < m && x < n)
                    tmp = std::max(tmp, image.at<uchar>(y, x));
            }
            dst[j] = tmp;
        }
    }

    return image_;
}

cv::Mat erosion(const cv::Mat &image, const Kernel &k) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            uchar tmp = UCHAR_MAX;
            for (auto &p : k) {
                int y{i + p[0]}, x{j + p[1]};
                if (y >= 0 && x >= 0 && y < m && x < n)
                    tmp = std::min(tmp, image.at<uchar>(y, x));
            }
            dst[j] = tmp;
        }
    }

    return image_;
}

cv::Mat opening(const cv::Mat &image, const Kernel &k) {
    return dilation(erosion(image, k), k);
}

cv::Mat closing(const cv::Mat &image, const Kernel &k) {
    return erosion(dilation(image, k), k);
}

int main() {
    cv::Mat image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};

    cv::Mat M;

    const Kernel k{octagonKernel()};
    M = dilation(image, k);
    cv::imwrite("dilation.bmp", M);

    M = erosion(image, k);
    cv::imwrite("erosion.bmp", M);

    M = opening(image, k);
    cv::imwrite("opening.bmp", M);

    M = closing(image, k);
    cv::imwrite("closing.bmp", M);

    return 0;
}