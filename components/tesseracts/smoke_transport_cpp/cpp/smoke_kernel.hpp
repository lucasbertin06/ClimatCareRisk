// Copyright 2026 ClimaCare-Risk contributors. SPDX-License-Identifier: Apache-2.0
//
// SmokeTransport: explicit advection-diffusion-decay solver and its discrete
// adjoint. Implements sections 5 and 6 of docs/mathematical_specification.md.
//
//   dc/dt + w . grad c = D_c laplacian(c) - lambda_c c + S,   c(x, 0) = 0
//
// One step is c^{n+1} = A(psi) c^n + dt S^n with
//   A(psi) = I + dt [ L_adv(w) + D_c L_lap - lambda_c I ].
//
// Boundary conditions are encoded explicitly and identically in the forward
// operator and in its transpose:
//   * advection uses a first-order upwind stencil with a homogeneous value
//     outside the domain on the inflow side; on the outflow side the same
//     one-sided stencil only reads interior cells, which is the discrete
//     convective outflow;
//   * diffusion uses a five-point centred Laplacian with zero normal flux,
//     implemented by clamped (replicated) neighbour access.
//
// The kernel is written in C++20 and parallelised with OpenMP over the slow
// spatial index. Reductions are accumulated per row and summed in index order,
// so results do not depend on the thread count.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace climacare {

struct TransportParams {
  int n_steps = 0;
  int ny = 0;
  int nx = 0;
  double wx = 0.0;
  double wy = 0.0;
  double diffusivity = 0.0;
  double decay = 0.0;
  double dt = 0.0;

  [[nodiscard]] double dx() const { return 1.0 / static_cast<double>(nx); }
  [[nodiscard]] double dy() const { return 1.0 / static_cast<double>(ny); }
  [[nodiscard]] std::size_t cells() const {
    return static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
  }
};

// Sufficient explicit-stability budget nu_c of section 5.4, computed from the
// actual wind components.
[[nodiscard]] double cfl_number(const TransportParams& params);

// Throws std::invalid_argument when any precondition of section 5.4 fails.
void validate(const TransportParams& params);

// Bilinear observation stencil of section 5.5. Throws when a sensor is too
// close to a boundary for the four-cell stencil to stay in range.
struct SensorStencil {
  std::vector<int> i0;
  std::vector<int> j0;
  std::vector<double> weights;  // [sensor][dj][di], row-major, 4 per sensor
  [[nodiscard]] int size() const { return static_cast<int>(i0.size()); }
};

[[nodiscard]] SensorStencil build_stencil(const TransportParams& params,
                                          const double* positions, int n_sensors);

// history must hold (n_steps + 1) * cells() doubles; history[0] is c^0 = 0.
void forward_history(const TransportParams& params, const double* source,
                     double* history);

// out must hold n_steps * n_sensors doubles: observations at levels 1..n_steps.
void observe(const TransportParams& params, const SensorStencil& stencil,
             const double* history, const double* bias, double* out);

// Frames of the concentration history at the requested history levels.
void extract_frames(const TransportParams& params, const double* history,
                    const int* frame_levels, int frame_count, double* out);

// Discrete adjoint of section 6. cotangent has shape (n_steps, n_sensors) and
// corresponds to the observations at levels 1..n_steps. source_bar must hold
// n_steps * cells() doubles; wind_bar holds two doubles.
void vector_jacobian_product(const TransportParams& params,
                             const SensorStencil& stencil, const double* source,
                             const double* cotangent, double* source_bar,
                             double* wind_bar, double* diffusivity_bar,
                             double* decay_bar);

}  // namespace climacare
