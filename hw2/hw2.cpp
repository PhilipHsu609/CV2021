#include <array>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <unordered_map>

#include "DisjointSet.h"

// [up, left, down, right, centroid_x, centroid_y]
using BBox = std::array<int, 6>;
using GrayscaleArray = std::array<int, 256>;

const cv::String Lena{"../lena.bmp"};

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

std::unordered_map<int, BBox> findBBox(std::vector<std::vector<int>> &label, std::unordered_map<int, int> &cc) {
    std::unordered_map<int, BBox> bbox;  // {componentID : BBox}

    // filter the connected components (cc) which have more than 500 pixels
    int validComponents = 0;
    for (auto &c : cc) {
        if (c.second >= 500) {
            validComponents++;
            bbox[c.first] = {512, 521, 0, 0, 0, 0};
        }
    }

    std::cout << "# of valid connected components: " << validComponents << std::endl;

    int m = label.size(), n = label[0].size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto it{bbox.find(label[i][j])};
            if (it != bbox.end()) {
                // updating bbox
                it->second[0] = std::min(it->second[0], i);
                it->second[1] = std::min(it->second[1], j);
                it->second[2] = std::max(it->second[2], i);
                it->second[3] = std::max(it->second[3], j);

                // (x, y) coordinate summation
                it->second[4] += j;
                it->second[5] += i;
            }
        }
    }

    // dividing the number of pixels of each component.
    for (auto &b : bbox) {
        b.second[4] /= cc[b.first];
        b.second[5] /= cc[b.first];
    }

    return bbox;
}

void draw(cv::Mat &image, const std::unordered_map<int, BBox> &bbox) {
    constexpr int thickness{3}, radius{5};
    for (auto &b : bbox) {
        cv::rectangle(
            image,
            cv::Point(b.second[1], b.second[0]),
            cv::Point(b.second[3], b.second[2]),
            cv::Scalar(255, 0, 0),
            thickness);
        // drawing at CENTROID
        cv::circle(
            image,
            cv::Point(b.second[4], b.second[5]),
            radius,
            cv::Scalar(0, 0, 255),
            cv::FILLED);
    }
}

cv::Mat connectedComponents(const cv::Mat &image) {
    int m = image.rows, n = image.cols;
    std::vector<std::vector<int>> label(m, std::vector<int>(n, 0));

    // first pass: labeling
    int counter = 0, targetLabel = 0;
    for (int i = 0; i < m; i++) {
        const uchar *row{image.ptr<uchar>(i)};
        for (int j = 0; j < n; j++) {
            if (row[j] != 0) {
                int up{i > 0 ? label[i - 1][j] : 0}, left{j > 0 ? label[i][j - 1] : 0};
                if (up == 0 && left == 0)
                    targetLabel = ++counter;
                else if (up == 0)
                    targetLabel = left;
                else if (left == 0)
                    targetLabel = up;
                else
                    targetLabel = std::min(up, left);
                label[i][j] = targetLabel;
            }
        }
    }

    std::cout << "Counter after first pass: " << counter << std::endl;

    // second pass: find equavilence classes with union find
    // using 4-connected
    DisjointSet ds(counter);
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            int up{label[i - 1][j]}, left{label[i][j - 1]}, cur{label[i][j]};
            if (cur != 0 && up != 0 && left != 0)
                if (ds.find_parent(up) != ds.find_parent(left))
                    ds.union_set(up, left);
        }
    }

    // third pass: assign component's label
    std::unordered_map<int, int> cc;  // {componentID : componentPixels}
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (label[i][j] != 0) {
                label[i][j] = ds.find_parent(label[i][j]);
                cc[label[i][j]]++;
            }
        }
    }

    cv::Mat image_;
    // convert image to RGB for drawing bounding boxes
    cv::cvtColor(image, image_, cv::COLOR_GRAY2BGR);
    draw(image_, findBBox(label, cc));

    return image_;
}

int main() {
    cv::Mat image{cv::imread(Lena, cv::IMREAD_GRAYSCALE)};
    std::cout << "Image size: " << image.size() << std::endl;

    GrayscaleArray freq{countFrequency(image)};
    writeCSV(freq, "freq.csv");

    cv::Mat M{binarize(image, 128)};
    cv::imwrite("binarize.bmp", M);

    cv::Mat label{connectedComponents(M)};
    cv::imwrite("label.bmp", label);

    return 0;
}