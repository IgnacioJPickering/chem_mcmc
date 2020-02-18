#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cmath>
#include <vector>

namespace py = pybind11;

class Bounds {
 public:
  std::string kind_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<double> sizes_;
  int dimension_;
  Bounds(const std::string &kind, const std::vector<double> &lower
         ,const std::vector<double> &upper) {
    lower_ = lower;
    upper_ = upper;
    kind_ = kind;
    dimension_ = int(lower_.size());
    for (int j=0; j != int(lower_.size()); j+=1){
      sizes_.push_back(upper_[j] - lower_[j]);
    }
  }

  bool areInBounds(const std::vector<double> &coordinates) {
    for (int j = 0; j != int(coordinates.size()); j += 1) {
      if (coordinates[j] < lower_[j] or coordinates[j] > upper_[j]) {
        return false;
      }
    }
    return true;
  }

  std::vector<double> wrapCoordinates(std::vector<double> coordinates) {
    for (int j = 0; j != int(coordinates.size()); j += 1) {
      if (coordinates[j] < lower_[j]) {
        coordinates[j] += sizes_[j];
      }
      if (coordinates[j] > upper_[j]) {
        coordinates[j] -= sizes_[j];
      }
    }
    return coordinates;
  }

  double getWrappedDistance(std::vector<double> &coordinates1, std::vector<double> &coordinates2){
    double sqdiffs = 0.;
    for (int j=0; j != int(coordinates1.size()); j+=1){
      double diff = coordinates1[j] - coordinates2[j];
      if (diff > sizes_[j]/2.){
        diff -= sizes_[j];
      }
      if (diff < -sizes_[j]/2.){
        diff += sizes_[j];
      }
      sqdiffs += std::pow(diff, 2.);
    }
    return std::sqrt(sqdiffs);
  }
};

// This is binding code for pybind11, usually binding code and
// implementation are in different files
PYBIND11_MODULE(staging, m) {
  m.doc() = "Bounds class implemented in cpp";  // optional module docstring
  py::class_<Bounds>(m, "Bounds")
      .def(py::init< const std::string &, const std::vector<double> &, const std::vector<double> &>())
      .def("areInBounds", &Bounds::areInBounds, py::arg("coordinates"))
      .def("getWrappedDistance", &Bounds::getWrappedDistance, py::arg("coordinates1"), py::arg("coordinates2"))
      .def("wrapCoordinates", &Bounds::wrapCoordinates, py::arg("difference"));
}
