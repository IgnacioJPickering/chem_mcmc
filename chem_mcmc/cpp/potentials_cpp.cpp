#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <math.h>
#include <vector>


double PI = 3.141592653589793;

double gaussian(double &r, double &sigma, double &center, double &height){
  double two_var = 2*std::pow(sigma, 2.);
  double constant = -1/(two_var);
  double exponent = constant*std::pow(r-center, 2.);
  return height*((1/sqrt(PI*two_var))*exp(exponent));
}; 

double log_gaussian(double &r, std::vector<double> &sigmas, std::vector<double> &A, std::vector<double> &mu){
  double sum_gaussians = 0.;
  for (int j=0; j < int(sigmas.size()); j += 1){
    double height = -A[j]*sqrt(PI * 2 * std::pow(sigmas[j],2.));
    sum_gaussians += gaussian(r, sigmas[j], height, mu[j]);
    }
  return -log(sum_gaussians);
}  



//This is binding code for pybind11, usually binding code and
//implementation are in different files
PYBIND11_MODULE(potentials_cpp, m) {
    m.doc() = "Potentials optimized in C++"; // optional module docstring
    m.def("gaussian", &gaussian, "Gaussian function");
    m.def("log_gaussian", &log_gaussian, "Log of the sum of Gaussian functions");
}
