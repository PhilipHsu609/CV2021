#include <opencv2/core.hpp>
#include <vector>

class Timer {
   public:
    Timer() : m_beg{static_cast<double>(cv::getTickCount())} {}

    void reset() {
        m_beg = static_cast<double>(cv::getTickCount());
    }

    double elapsed() const {
        return (static_cast<double>(cv::getTickCount()) - m_beg) / cv::getTickFrequency();
    }

   private:
    double m_beg;
};

template <class T>
void writeCSV(const std::vector<std::vector<T>> &v, const cv::String &fileName) {
    std::ofstream os{fileName, std::ios::out};
    std::stringstream ss;

    if (!os.is_open()) {
        std::cerr << "Open file failed..." << std::endl;
        exit(-1);
    }

    for (auto &row : v) {
        for (auto &val : row) {
            ss << val << ',';
        }
        ss << '\n';
    }

    os << ss.str();

    os.close();
}