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

double lennard_jones(const double r, const double epsilon, const double sigma, const double center, const double delta, const double cutoff){
  //By default center = 0.0 and delta = 1e-10
  if (r > cutoff) {
      return 0.;
  }
  double term1 = sigma / std::pow(r - center + delta, 6.);
  double term2 = term1 * term1;
  return 4 * epsilon * (term2 - term1);
}

double lennard_jones_dv(const double r, const double epsilon, const double sigma, const double center, const double delta, const double cutoff){
  //By default center = 0.0 and delta = 1e-10
  if (r > cutoff) {
      return 0.;
  }
  double term1 = 6 * std::pow(sigma, 6.) / std::pow(r - center + delta, 7.);
  double term2 = 12 * std::pow(sigma, 12.) / std::pow(r - center + delta, 13.);
  return 4 * epsilon * (- term2 + term1);
}


// This is binding code for pybind11, usually binding code and
// implementation are in different files
PYBIND11_MODULE(potentials_cpp, m) {
  m.doc() = "Potentials optimized in C++";  // optional module docstring
  m.def("gaussian", &gaussian, "Gaussian function");
  m.def("log_gaussian", &log_gaussian, "Log of the sum of Gaussian functions");
  m.def("lennard_jones", &lennard_jones, "Lennard jones function, using the epsilon-sigma parametrization");
  m.def("lennard_jones_dv", &lennard_jones, "Lennard jones derivative function function, using the epsilon-sigma parametrization");
}
