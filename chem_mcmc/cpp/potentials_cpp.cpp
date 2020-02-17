#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <vector>


double PI = 3.141592653589793;

double gaussian(const double &r, const double &sigma, const double &center,
                const double &height){
  double two_var = 2 * std::pow(sigma, 2.);
  double constant = -1 / (two_var);
  double exponent = constant * std::pow(r - center, 2.);
  return -height * ((1 / std::sqrt(PI * two_var)) * std::exp(exponent));
};

double log_gaussian(const double &r, const std::vector<double> &sigmas,
                    const std::vector<double> &A,
                    const std::vector<double> &mu){
  double sum_gaussians = 0.;
  for (int j = 0; j != int(sigmas.size()); j += 1) {
    double two_var = 2 * std::pow(sigmas[j], 2.);
    double constant = -1 / (two_var);
    double exponent = constant * std::pow(r - mu[j], 2.);
    sum_gaussians += A[j] * std::exp(exponent);
  }
  return -std::log(sum_gaussians);
}

// This is binding code for pybind11, usually binding code and
// implementation are in different files
PYBIND11_MODULE(potentials_cpp, m) {
  m.doc() = "Potentials optimized in C++";  // optional module docstring
  m.def("gaussian", &gaussian, "Gaussian function");
  m.def("log_gaussian", &log_gaussian, "Log of the sum of Gaussian functions");
}
