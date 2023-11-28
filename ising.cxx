#include <NumCpp.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>

using namespace nc;

// 2D coordinates representing a site in the spin lattice S.
#define SITE s[0], s[1]

// Spin lattice size
#define SZ N * N

// von Neumann neighborhood
#define n1 (s[0] + 1) % N, s[1]
#define n2 s[0], (s[1] + 1) % N
#define n3 (s[0] - 1) % N, s[1]
#define n4 s[0], (s[1] - 1) % N

// *** Global simulation constants ***

// Data columns (for writing to file)
enum DATA_COLS { TEMP, ORDER, CHI, CB, U };

// Simulation parameters
const unsigned int M = 20000;
const unsigned int therm = 1000;
const unsigned int MC = 1;
const std::string method1 = "metropolis";
const std::string method2 = "heatbath";

// Temperature steps
struct {
  const double max = 5;
  const double min = 0.1;
  const double steps = 80;
} T;

// Measurements
const int n_meas = 5;

// ************************************

NdArray<double> ising(const int N, const double J, const double B,
                      const std::string method_opt);

// Metropolis-Hastings
void metropolis(NdArray<int> &S, const int N, double kBT_i, const double J,
                const double B);

// Heat-bath
void heatbath(NdArray<int> &S, const int N, double kBT_i, const double J);

// Print data columns
void tofile(NdArray<double> &measurements, std::string file);

int main(int argc, char *argv[]) {

  const auto input_method = std::string(argv[1]);

  if (argc != 6) {
    fprintf(stderr,
            "USAGE: <metropolis/heatbath> <lattice dim> <J> <B> <filename>\n");
    return EXIT_FAILURE;
  } else if (!(input_method == method1 || input_method == method2)) {
    fprintf(stderr, "Methods: <metropolis/heatbath>\n");
    return EXIT_FAILURE;
  }

  // Input args
  const int N = atoi(argv[2]);      // Square lattice dims
  const double J = atof(argv[3]);   // Coupling strength
  const double B = atof(argv[4]);   // External B-field strength
  const std::string file = argv[5]; // File to store data in

  // Print columns
  NdArray<double> data = ising(N, J, B, input_method);
  tofile(data, file);
  return 0;
}

// DONE function pointer for choosing method
NdArray<double> ising(const int N, const double J, const double B,
                      const std::string method_opt) {

  // Set random seed for reproducibility
  random::seed(1337);

  double m0 = 0, m = 0, E0 = 0, E = 0, E2 = 0, m2 = 0, m4 = 0;

  // Temperature from high to low
  NdArray<double> kBT = fliplr(linspace(T.min, T.max, T.steps));

  // Function to handle computation
  // Simplified to using only variables in signature
  // Mostly made for learning
  std::function<void(NdArray<int> &, double)> computation;

  // The two methods are checked during input before
  // getting bound to constant args. 
  if (method_opt == method1) {
  //f(a,b,c,d,e) -> [b,c,e]f(a,d)
    computation = std::bind(&metropolis, std::placeholders::_1, N,
                            std::placeholders::_2, J, B);
  } else {
  //f(a,b,c,d) -> [b,d]f(a,c)
    computation = std::bind(&heatbath, std::placeholders::_1, N,
                            std::placeholders::_2, J);
  }

  // Order parameter lambda function
  auto order = [N](NdArray<int> &S) {
    return abs(sum<int>(S).item()) / ((double)SZ);
  };

  // Energy lambda function
  auto energy = [N, J, B](NdArray<int> &S) {
    NdArray<int> nbrs = roll(S, 1, Axis::COL) + roll(S, -1, Axis::COL) +
                roll(S, 1, Axis::ROW) + roll(S, -1, Axis::ROW);

    return (-J * sum(matmul(S, nbrs)).item() - B * sum(S).item()) /
           ((double)SZ);
  };

  // Init spin matrix with random ICs
  auto S = random::choice<int>({-1, 1}, SZ).reshape(N, N);

  // Measurement matrix
  auto measurements = NdArray<double>(kBT.shape().cols, n_meas);

  for (size_t T_i = 0; T_i < kBT.shape().cols; T_i++) {

    // Thermalization
    for (size_t i = 0; i < therm * SZ; i++) {
      computation(S, kBT[T_i]);
    }

    // Reset measurements
    m = m2 = m4 = E = E2 = 0;

    // Simulation and measurements
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < MC * SZ; j++) {
        computation(S, kBT[T_i]);
      }

      // Order param
      m0 = order(S);
      // Energy
      E0 = energy(S);

      // Accumulate
      m += m0;
      m2 += m0 * m0;
      m4 += m0 * m0 * m0 * m0;
      E += E0;
      E2 += E0 * E0;
    }

    // Average over M samples
    m = m / M;
    m2 = m2 / M;
    m4 = m4 / M;
    E = E / M;
    E2 = E2 / M;

    // Add to measurements
    measurements(T_i, TEMP) = kBT[T_i];
    measurements(T_i, ORDER) = m;
    measurements(T_i, CHI) = (m2 - m * m) / (kBT[T_i]);
    measurements(T_i, CB) = (E2 - E * E) / (kBT[T_i] * kBT[T_i]);
    measurements(T_i, U) = 1 - m4 / (3 * (m2 * m2));
  }
  return measurements;
}

// Metropolis-Hastings (implemented with external field)
void metropolis(NdArray<int> &S, const int N, double kBT_i, const double J,
                const double B) {
  NdArray<int> s = random::randInt({1, 2}, N);
  int S_alpha_beta = S(SITE) * (S(n1) + S(n2) + S(n3) + S(n4));
  double dE = 2.0 * J * S_alpha_beta + 2.0 * B * S(SITE);
  if (dE <= 0 || random::rand<double>() < exp(-dE / kBT_i))
    S(SITE) = -S(SITE);
}

// Heat-bath (not implemented with external field)
void heatbath(NdArray<int> &S, const int N, double kBT_i, const double J) {
  NdArray<int> s = random::randInt({1, 2}, N);
  int s_j = S(n1) + S(n2) + S(n3) + S(n4);
  double p_i = 1.0 / (1.0 + exp(-2.0 * J * s_j / kBT_i));
  // Always accept the change
  S(SITE) = (random::rand<double>() < p_i) ? 1 : -1;
}

// Write columns to file (NumCPP has a tofile() function that can be reshaped
// during the data analysis)
void tofile(NdArray<double> &measurements, std::string file) {
  std::ofstream out;
  std::string path = "../data/";
  out.open(path + file);
  // Header
  out << "temp "
      << "order "
      << "chi "
      << "cb "
      << "u" << std::endl;
  // Data
  for (size_t i = 0; i < measurements.shape().rows; i++) {
    for (size_t j = 0; j < measurements.shape().cols; j++) {
      out << measurements(i, j) << ' ';
    }
    out << std::endl;
  }
}
