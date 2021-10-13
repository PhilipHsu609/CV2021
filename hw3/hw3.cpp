#include <array>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

using GrayscaleArray = std::array<int, 256>;
const cv::String Lena{"../lena.bmp"};

GrayscaleArray countFrequency(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    GrayscaleArray freq;

    freq.fill(0);
    for (int i = 0; i < m; i++) {
        const uchar *p{image.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            freq[p[j]]++;
        }
    }

    return freq;
}

void writeCSV(const GrayscaleArray &hist, const cv::String &fileName) {
    std::ofstream os{fileName, std::ios::out};
    std::stringstream ss;

    if (!os.is_open()) {
        std::cerr << "Open file failed..." << std::endl;
        exit(-1);
    }

    ss << "m,h(m)\n";
    for (int i = 0; i < hist.size(); i++) {
        ss << i << ',' << hist[i] << '\n';
    }

    os << ss.str();

    os.close();
}

cv::Mat lowerIntensity(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            dst[j] = src[j] / 3;
        }
    }

    return image_;
}

cv::Mat histogramEqualization(const cv::Mat &image, const GrayscaleArray &freq) {
    int m = image.rows, n = image.cols;
    double N = m * n;
    std::vector<double> T(freq.size(), 0);
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));

    T[0] = freq[0] / N;
    for (int i = 1; i < freq.size(); i++) {
        T[i] = T[i - 1] + freq[i] / N;
    }

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            dst[j] = 255 * T[src[j]];
        }
    }

    return image_;
}

int main() {
    cv::Mat image{cv::imread(Lena, cv::IMREAD_GRAYSCALE)}, M;
    GrayscaleArray f;

    f = countFrequency(image);
    writeCSV(f, "a.csv");

    M = lowerIntensity(image);
    cv::imwrite("b.bmp", M);

    f = countFrequency(M);
    writeCSV(f, "b.csv");

    M = histogramEqualization(M, f);
    cv::imwrite("c.bmp", M);

    f = countFrequency(M);
    writeCSV(f, "c.csv");

    return 0;
}