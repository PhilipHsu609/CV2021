#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using Kernel = std::vector<std::vector<int>>;

const cv::String lena{"../lena.bmp"};
const Kernel J{{0, -1}, {0, 0}, {1, 0}};
const Kernel K{{-1, 0}, {-1, 1}, {0, 1}};

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

cv::Mat binarize(const cv::Mat &image, int threshold) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            dst[j] = (src[j] >= threshold) ? 255 : 0;
        }
    }

    return image_;
}

cv::Mat dilation(const cv::Mat &image, const Kernel &k) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            if (src[j] == 0xFF) {
                bool keep = false;
                for (auto &p : k) {
                    int y{i + p[0]}, x{j + p[1]};
                    if (y == i && x == j)
                        keep = true;
                    if (x >= 0 && y >= 0 && x < n && y < m)
                        image_.at<uchar>(y, x) = 0xFF;
                }
                image_.at<uchar>(i, j) = keep ? 0xFF : 0;
            }
        }
    }

    return image_;
}

cv::Mat erosion(const cv::Mat &image, const Kernel &k) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            bool keep = true;
            for (int p = 0; p < k.size() && keep; p++) {
                int y{i + k[p][0]}, x{j + k[p][1]};
                if (x >= 0 && y >= 0 && x < n && y < m)
                    keep = image.at<uchar>(y, x) != 0;
            }
            image_.at<uchar>(i, j) = keep ? 0xFF : 0x00;
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

cv::Mat complement(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            dst[j] = (src[j] == 0xFF) ? 0 : 0xFF;
        }
    }

    return image_;
}

cv::Mat intersect(const cv::Mat &image1, const cv::Mat &image2) {
    if (image1.rows != image2.rows || image1.cols != image2.cols) {
        std::cerr << "Intersect need two cv::Mat with same shape.\n";
        exit(-1);
    }

    int m = image1.rows, n = image1.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *p1{image1.ptr<uchar>(i)}, *p2{image2.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            dst[j] = (p1[j] == p2[j]) ? p1[j] : 0;
        }
    }

    return image_;
}

cv::Mat hitAndMiss(const cv::Mat &image, const Kernel &j, const Kernel &k) {
    cv::Mat hit, miss;
    hit = erosion(image, j);
    miss = erosion(complement(image), k);
    return intersect(hit, miss);
}

int main() {
    cv::Mat image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};
    image = binarize(image, 128);

    const Kernel k{octagonKernel()};

    cv::Mat M;
    M = dilation(image, k);
    cv::imwrite("dilation.bmp", M);

    M = erosion(image, k);
    cv::imwrite("erosion.bmp", M);

    M = opening(image, k);
    cv::imwrite("opening.bmp", M);

    M = closing(image, k);
    cv::imwrite("closing.bmp", M);

    M = hitAndMiss(image, J, K);
    cv::imwrite("hit-and-miss.bmp", M);

    return 0;
}