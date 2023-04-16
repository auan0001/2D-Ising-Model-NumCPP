# 2D Ising Model NumCPP Implementation
## Running the simulations
This is a Markov Chain Monte Carlo simulation of the 2D Ising Model in C++. The dependencies are NumCPP and CMake (for building, using GCC).
Build and compile the project by changing the the build directory

```
% cd build
% cmake ..
```

Running the binary with parameters $L=32$, $J=1$, $B=0$ and saving the results in `run.dat`

```
./ISING 32 1.0 0.0 run.dat
```
If number of args are incorrect, the console will display

```
USAGE: <lattice dim> <J> <B> <file>
```

## Results
These plots are excerpts generated from the simulation data
### Spin lattice configurations over different temperatures
![alt text](https://github.com/auan0001/2D-Ising-Model-NumCPP/blob/main/images/lattice3.png)
### Varying the field with a negative spin coupling
![alt text](https://github.com/auan0001/2D-Ising-Model-NumCPP/blob/main/images/varying.png)
