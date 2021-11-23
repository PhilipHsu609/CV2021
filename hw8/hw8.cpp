#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <vector>

using Kernel = std::vector<std::vector<int>>;

// const int seed = 11533;
const cv::String lena{"../lena.bmp"};
std::mt19937 mersenne{static_cast<std::mt19937::result_type>(std::time(nullptr))};

cv::Mat addGaussianNoise(const cv::Mat &image, int amplitude) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));
    std::normal_distribution<double> N(0, 1);

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            int noisePixel{src[j] + amplitude * static_cast<int>(N(mersenne))};
            if (noisePixel < 0) noisePixel = 0;
            if (noisePixel > 255) noisePixel = 255;
            dst[j] = noisePixel;
        }
    }

    return image_;
}

cv::Mat addSaltAndPepperNoise(const cv::Mat &image, double threshold) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));
    std::uniform_real_distribution<double> u(0, 1);

    auto saltOrPepper{[&]() -> int {
        double sample{u(mersenne)};
        if (sample < threshold) {
            return 0;
        }
        if (sample > 1 - threshold) {
            return 255;
        }
        return -1;
    }};

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            int choice{saltOrPepper()};
            dst[j] = (choice == -1) ? src[j] : choice;
        }
    }

    return image_;
}

cv::Mat boxFilter(const cv::Mat &image, int kernelSize) {
    int m = image.rows, n = image.cols, k = kernelSize / 2;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));
    cv::Mat pad(m + 2 * k, n + 2 * k, CV_8UC1, cv::Scalar::all(0));
    cv::copyMakeBorder(image, pad, k, k, k, k, cv::BORDER_REPLICATE);

    for (int i = 0; i < m; i++) {
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            int acc = 0;

            for (int ii = -k; ii <= k; ii++) {
                for (int jj = -k; jj <= k; jj++) {
                    acc += pad.at<uchar>(i + k + ii, j + k + jj);
                }
            }

            dst[j] = acc / (kernelSize * kernelSize);
        }
    }

    return image_;
}

cv::Mat medianFilter(const cv::Mat &image, int kernelSize) {
    int m = image.rows, n = image.cols, k = kernelSize / 2;
    cv::Mat image_(m, n, CV_8UC1, cv::Scalar::all(0));
    cv::Mat pad(m + 2 * k, n + 2 * k, CV_8UC1, cv::Scalar::all(0));
    cv::copyMakeBorder(image, pad, k, k, k, k, cv::BORDER_REPLICATE);

    for (int i = 0; i < m; i++) {
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            std::vector<uchar> tmp;
            for (int ii = -k; ii <= k; ii++) {
                for (int jj = -k; jj <= k; jj++) {
                    tmp.push_back(pad.at<uchar>(i + k + ii, j + k + jj));
                }
            }
            std::sort(begin(tmp), end(tmp));
            dst[j] = tmp[tmp.size() / 2];
        }
    }

    return image_;
}

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

double SNR(const cv::Mat &result, const cv::Mat &noise) {
    int m = result.rows, n = result.cols;
    double mu_s = 0, mu_n = 0, vs = 0, vn = 0;

    for (int i = 0; i < m; i++) {
        const uchar *I{result.ptr<uchar>(i)}, *I_n{noise.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            mu_s += I[j] / 255.0;
            mu_n += (I_n[j] - I[j]) / 255.0;
        }
    }
    mu_s /= m * n;
    mu_n /= m * n;

    for (int i = 0; i < m; i++) {
        const uchar *I{result.ptr<uchar>(i)}, *I_n{noise.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            vs += std::pow(I[j] / 255.0 - mu_s, 2);
            vn += std::pow((I_n[j] - I[j]) / 255.0 - mu_n, 2);
        }
    }
    vs /= m * n;
    vn /= m * n;

    return 10 * std::log10(vs / vn);
}

int main() {
    Kernel k{octagonKernel()};
    cv::Mat image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};
    cv::Mat M;

    std::vector<cv::String> noiseName{"g10", "g30", "sap005", "sap010"};
    std::vector<cv::String> suffix{"_box3", "_box5", "_med3", "_med5", "_oc", "_co"};
    std::vector<cv::Mat> noiseImage{
        addGaussianNoise(image, 10),
        addGaussianNoise(image, 30),
        addSaltAndPepperNoise(image, 0.05),
        addSaltAndPepperNoise(image, 0.1)};
    std::vector<cv::Mat> resultImage;

    for (int i = 0; i < 4; i++) {
        cv::String output{noiseName[i] + ".bmp"};
        cv::imwrite(output, noiseImage[i]);
        std::cout << "Current file: " << output << "\t SNR = " << SNR(image, noiseImage[i]) << std::endl;
        resultImage.push_back(boxFilter(noiseImage[i], 3));
        resultImage.push_back(boxFilter(noiseImage[i], 5));
        resultImage.push_back(medianFilter(noiseImage[i], 3));
        resultImage.push_back(medianFilter(noiseImage[i], 5));
        resultImage.push_back(closing(opening(noiseImage[i], k), k));
        resultImage.push_back(opening(closing(noiseImage[i], k), k));
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++) {
            cv::String output{noiseName[i] + suffix[j] + ".bmp"};
            std::cout << "Current file: " << output << "\t SNR = " << SNR(image, resultImage[i * 6 + j]) << std::endl;
            cv::imwrite(output, resultImage[i * 6 + j]);
        }
    }

    return 0;
}