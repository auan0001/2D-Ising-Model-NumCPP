#include <NumCpp.hpp>
#include <cstddef>
#include <iostream>
#include <string>

// Indexing, normalizing and averages
#define SITE s[0],s[1]
#define SZ N*N
#define n1 (s[0]+1)%N, s[1]
#define n2 s[0], (s[1]+1)%N
#define n3 (s[0]-1)%N, s[1]
#define n4 s[0], (s[1]-1)%N
#define idx(dH) (dH+4)/2
#define avgnorm(x) x/((double)M*N*N)
#define norm(x) x/((double)N*N)

// Metropolis-Hastings
void metropolis(nc::NdArray<int>& S, const int N, nc::NdArray<double>& kBT, const double J, const double B, double T_i);

// Heat-bath
void heatbath(nc::NdArray<int>& S, const int N, nc::NdArray<double>& kBT, const double J, double T_i);

// Energy
double energy(nc::NdArray<int>& S, const int N, const double J, const double B);

// Print data columns
void tofile(nc::NdArray<double>& measurements, std::string file);

int main (int argc, char *argv[]) {
  if (argc < 5 || argc >= 6) {
    std::cout << std::endl << "USAGE: <lattice dim> <J> <B> <file>" << std::endl;
    return EXIT_FAILURE;
  }

  // Input args
  const int N = atoi(argv[1]);      // Square lattice dims
  const double J = atof(argv[2]);   // Coupling strength
  const double B = atof(argv[3]);   // External B-field strength
  const std::string file = argv[4]; // File to store data in

  // Data columns (for writing to file)
  enum DATA_COLS {TEMP, ORDER, CHI, CB, U};

  // Simulation parameters
  const unsigned int M = 200000; 
  const unsigned int therm = 1000;
  const unsigned int MC = 1;

  // Temperature steps
  struct {
    const double max = 5;
    const double min = 0.1;
    const double step = 80;
  } T;

  // Set random seed for reproducibility
  nc::random::seed(1337);

  // Measurements
  const int n_meas = 5;
  double m0 = 0,
         m = 0,
         E0 = 0,
         E = 0,
         E2 = 0,
         m2 = 0,
         m4 = 0;

  // Temperature from high to low
  auto kBT = nc::fliplr(nc::linspace(T.min, T.max, T.step));

  // Order parameter lambda
  auto order = [](auto S, auto N){
    return nc::abs(nc::sum<int>(S).item())/((double)SZ);
  };

  // Init spin matrix with random ICs
  auto S = nc::random::choice<int>({-1,1},SZ).reshape(N,N);

  // Measurement matrix
  auto measurements = nc::NdArray<double>(kBT.shape().cols,n_meas);

  for (size_t T_i = 0; T_i < kBT.shape().cols; T_i++) {

    // Thermalization
    for (size_t i = 0; i < therm*SZ; i++) {
      // metropolis(S, N, kBT, J, B, T_i);
      heatbath(S, N, kBT, J, T_i);
    }

    // Reset measurements
    m = m2 = m4 = E = E2 = 0;

    // Simulation and measurements
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < MC*SZ; j++) {
        // metropolis(S, N, kBT, J, B, T_i);
        heatbath(S, N, kBT, J, T_i);
      }

      // Order param
      m0 = order(S,N);
      // Energy
      E0 = energy(S,N,J,B);

      // Accumulate
      m += m0;
      m2 += m0*m0;
      m4 += m0*m0*m0*m0;
      E += E0;
      E2 += E0*E0;
    }

    // Average over M samples
    m = m/M;
    m2 = m2/M;
    m4 = m4/M;
    E = E/M;
    E2 = E2/M;

    // Add to measurements
    measurements(T_i,TEMP) = kBT[T_i];
    measurements(T_i,ORDER) = m;
    measurements(T_i,CHI) = (m2-m*m)/(kBT[T_i]);
    measurements(T_i,CB) = (E2-E*E)/(kBT[T_i]*kBT[T_i]);
    measurements(T_i,U) = 1-m4/(3*(m2*m2));

  }

  // Print columns
  tofile(measurements, file);
  return 0;
}

// Metropolis-Hastings (implemented with external field)
void metropolis(nc::NdArray<int>& S, const int N, nc::NdArray<double>& kBT, const double J,  const double B, double T_i) {
  nc::NdArray<int> s = nc::random::randInt({1,2},N);
  auto S_alpha_beta = S(SITE)*(S(n1)+S(n2)+S(n3)+S(n4));
  auto dE = 2.0*J*S_alpha_beta+2.0*B*S(SITE);
  if (dE <= 0 || nc::random::rand<double>() < nc::exp(-dE/kBT[T_i])) 
    S(SITE) = -S(SITE);
}

// Heat-bath (not implemented with external field)
void heatbath(nc::NdArray<int>& S, const int N, nc::NdArray<double>& kBT, const double J, double T_i) {
  nc::NdArray<int> s = nc::random::randInt({1,2},N);
  auto s_j = S(n1)+S(n2)+S(n3)+S(n4);
  auto p_i = 1.0/(1.0 + nc::exp(-2.0*J*s_j/kBT[T_i]));
  // Always accept the change
  S(SITE) = (nc::random::rand<double>() < p_i) ? 1: -1;
}

// Energy
double energy(nc::NdArray<int>& S, const int N, const double J, const double B) {
  auto nbrs = nc::roll(S, 1, nc::Axis::COL)
    + nc::roll(S, -1, nc::Axis::COL)
    + nc::roll(S, 1, nc::Axis::ROW)
    + nc::roll(S, -1, nc::Axis::ROW);
  return (-J*nc::sum(nc::matmul(S,nbrs)).item() -B*nc::sum(S).item())/((double)SZ);
}

// Write columns to file (NumCPP has a tofile() function that can be reshaped during the data analysis)
void tofile(nc::NdArray<double>& measurements, std::string file) {
  std::ofstream out;
  std::string path = "./Data/";
  out.open(path+file);
  // Header
  out << "temp " << "order " << "chi " << "cb " << "u" << std::endl;
  // Data
  for (size_t i = 0; i < measurements.shape().rows; i++) {
    for (size_t j = 0; j < measurements.shape().cols; j++) {
      out << measurements(i,j) << ' ';
    }
    out << std::endl;
  }
}
