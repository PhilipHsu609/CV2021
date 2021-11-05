#include <functional>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

const cv::String lena{"../lena.bmp"};

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

cv::Mat downsample(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m / 8, n / 8, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m / 8; i++) {
        const uchar *src{image.ptr<uchar>(i * 8)};
        uchar *dst{image_.ptr<uchar>(i)};
        for (int j = 0; j < n / 8; j++) {
            dst[j] = src[j * 8];
        }
    }

    return image_;
}

char h_yokoi(int b, int c, int d, int e) {
    if (b != c) return 's';
    if (b == d && b == e) return 'r';
    return 'q';
}

int f_yokoi(const std::vector<char> &a) {
    int q_count = 0, r_count = 0;
    for (char c : a) {
        r_count += (c == 'r') ? 1 : 0;
        q_count += (c == 'q') ? 1 : 0;
    }
    return (r_count == 4) ? 5 : q_count;
}

char h_shrink(int b, int c, int d, int e) {
    if (b == c && (b != d || b != e)) return '1';
    return '0';
}

int f_shrink(const std::vector<char> &a, int x) {
    int a_count = 0;
    for (char c : a) {
        a_count += (c == '1') ? 1 : 0;
    }
    return (a_count == 1) ? 0 : x;
}

std::vector<char> neighbor(const cv::Mat &image, int row, int col, std::function<char(int, int, int, int)> h) {
    int m = image.rows, n = image.cols;
    std::vector<std::vector<std::vector<int>>> dirs{
        {{0, 0}, {0, 1}, {-1, 1}, {-1, 0}},    // a1
        {{0, 0}, {-1, 0}, {-1, -1}, {0, -1}},  // a2
        {{0, 0}, {0, -1}, {1, -1}, {1, 0}},    // a3
        {{0, 0}, {1, 0}, {1, 1}, {0, 1}}};     // a4
    std::vector<char> a;

    for (auto &dir : dirs) {
        char c = h(image.at<uchar>(row + dir[0][0], col + dir[0][1]),
                   image.at<uchar>(row + dir[1][0], col + dir[1][1]),
                   image.at<uchar>(row + dir[2][0], col + dir[2][1]),
                   image.at<uchar>(row + dir[3][0], col + dir[3][1]));
        a.push_back(c);
    }

    return a;
}

cv::Mat padding(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    cv::Mat image_(m + 2, n + 2, CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < m; i++) {
        const uchar *src{image.ptr<uchar>(i)};
        uchar *dst{image_.ptr<uchar>(i + 1)};
        for (int j = 0; j < n; j++) {
            dst[j + 1] = src[j];
        }
    }

    return image_;
}

std::vector<std::vector<int>> Yokoi(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    std::vector<std::vector<int>> label(m, std::vector<int>(n, 0));
    cv::Mat image_{padding(image)};

    for (int i = 0; i < m; i++) {
        const uchar *src{image_.ptr<uchar>(i + 1)};
        for (int j = 0; j < n; j++) {
            if (src[j + 1] != 0)
                label[i][j] = f_yokoi(neighbor(image_, i + 1, j + 1, h_yokoi));
        }
    }

    return label;
}

std::vector<std::vector<char>> pairRelation(std::vector<std::vector<int>> &label) {
    int m = label.size(), n = label[0].size();
    std::vector<std::vector<char>> marked(m, std::vector<char>(n, ' '));
    std::vector<std::vector<int>> dirs{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (label[i][j] != 0) {
                int one_count = 0;
                for (auto &dir : dirs) {
                    int y{i + dir[0]}, x{j + dir[1]};
                    if (y >= 0 && x >= 0 && y < m && x < n) {
                        one_count += (label[y][x] == 1) ? 1 : 0;
                    }
                }
                marked[i][j] = (one_count >= 1 && label[i][j] == 1) ? 'p' : 'q';
            }
        }
    }

    return marked;
}

cv::Mat connectedShrink(const cv::Mat &image, std::vector<std::vector<char>> &marked, bool &flag) {
    int m = image.rows, n = image.cols;
    cv::Mat image_ = padding(image);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            uchar *src{image_.ptr<uchar>(i + 1)};
            if (marked[i][j] == 'p') {
                int res = f_shrink(neighbor(image_, i + 1, j + 1, h_shrink), src[j + 1]);
                if (res != src[j + 1]) {
                    src[j + 1] = res;
                    flag = true;
                }
            }
        }
    }

    return image_(cv::Rect(1, 1, m, n));
}

cv::Mat thinning(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    bool change = true;
    cv::Mat image_;
    image.copyTo(image_);

    while (change) {
        change = false;
        auto label = Yokoi(image_);
        auto marked = pairRelation(label);
        image_ = connectedShrink(image_, marked, change);
    }

    return image_;
}

int main() {
    cv::Mat image{cv::imread(lena, cv::IMREAD_GRAYSCALE)};
    cv::Mat M;

    M = binarize(image, 128);
    M = downsample(M);
    M = thinning(M);

    cv::imwrite("thinning.bmp", M);

    return 0;
}