#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

const cv::String Input_image{"../lena.bmp"};

cv::Mat upsideDown(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i{}; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(m - i - 1)};
        for (int j{}; j < n; j++) {
            dst[j] = src[j];
        }
    }

    return image_;
}

cv::Mat rightSideLeft(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i{}; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j{}; j < n; j++) {
            dst[j] = src[n - j - 1];
        }
    }

    return image_;
}

cv::Mat diagonallyFlip(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i{}; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        for (int j{}; j < n; j++) {
            image_.at<uchar>(j, i) = src[j];
        }
    }

    return image_;
}

cv::Mat rotate(const cv::Mat &image, double theta) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    cv::Mat r = cv::getRotationMatrix2D(cv::Point2f(n / 2, m / 2), theta, 1.0);
    cv::warpAffine(image, image_, r, image.size());

    return image_;
}

cv::Mat shrinkHalf(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m / 2, n / 2, CV_8UC1, cv::Scalar::all(0));

    cv::resize(image, image_, cv::Size(m / 2, n / 2), 0.5, 0.5);

    return image_;
}

cv::Mat binarize(const cv::Mat &image, int threshold) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i{}; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j{}; j < n; j++) {
            dst[j] = (src[j] > threshold) ? 255 : 0;
        }
    }

    return image_;
}

int main() {
    cv::Mat Image = cv::imread(Input_image, cv::IMREAD_GRAYSCALE);
    std::cout << "Image size: " << Image.size() << std::endl;

    cv::Mat M;

    M = upsideDown(Image);
    cv::imwrite("lena_upsidedown.bmp", M);

    M = rightSideLeft(Image);
    cv::imwrite("lena_leftsideright.bmp", M);

    M = diagonallyFlip(Image);
    cv::imwrite("lena_diagonally.bmp", M);

    M = rotate(Image, -45);
    cv::imwrite("lena_rotate.bmp", M);

    M = shrinkHalf(Image);
    cv::imwrite("lena_shrink.bmp", M);

    M = binarize(Image, 128);
    cv::imwrite("lena_binarize.bmp", M);

    return 0;
}