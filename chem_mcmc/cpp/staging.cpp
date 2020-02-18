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
  Bounds(const std::string &kind="n", const std::vector<double> &lower={0.}
         ,const std::vector<double> &upper={10.}) {
    lower_ = lower;
    upper_ = upper;
    kind_ = kind;
    dimension_ = int(lower_.size());
    for (int j=0; j != int(lower_.size()); j+=1){
      sizes_.push_back(upper_[j] - lower_[j]);
    }
  }
  //Factory method
  static Bounds square(double lower=0.0, double upper=0.0, std::string kind="n", int dimension=1){
    std::vector<double> lower_vec;
    std::vector<double> upper_vec;
    for (int j=0;j != dimension; j++){
      lower_vec.push_back(lower);
      upper_vec.push_back(upper);
    }
    return Bounds(kind, lower_vec, upper_vec);
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
// Basically the ONLY thing I have to write is the PYBIND11 binding code, 
// The rest of the code is normal Cpp code, 100% normal
// Keep in mind that the pybind namespace defines some easy utilities
// for interfacing with C++ such as py::none and others
// def_readonly and def_readwrite are used for attributes
// def is used for functions and def_static for static methods (classmethods
// in python)
// Names can be changed for usage in python, which can be very useful 
// to keep python conventions!
PYBIND11_MODULE(staging, m) {
  m.doc() = "Bounds class implemented in cpp";  // optional module docstring
  py::class_<Bounds>(m, "Bounds")
      .def(py::init< const std::string &, const std::vector<double> &, const std::vector<double> &>())
      .def_readonly("kind", &Bounds::kind_)
      .def_readonly("lower", &Bounds::lower_)
      .def_readonly("upper", &Bounds::upper_)
      .def_readonly("sizes", &Bounds::sizes_)
      .def_readonly("dimension", &Bounds::dimension_)
      .def("are_in_bounds", &Bounds::areInBounds, py::arg("coordinates"))
      .def("get_wrapped_distance", &Bounds::getWrappedDistance, py::arg("coordinates1"), py::arg("coordinates2"))
      .def("wrap_coordinates", &Bounds::wrapCoordinates, py::arg("difference"))
      .def_static("square", &Bounds::square, py::arg("lower")=0.0, py::arg("upper")=0.0, py::arg("kind")="n", py::arg("dimension")=1);
}
