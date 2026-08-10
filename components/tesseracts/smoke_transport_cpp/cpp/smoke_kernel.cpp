// Copyright 2026 ClimaCare-Risk contributors. SPDX-License-Identifier: Apache-2.0
#include "smoke_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace climacare {
namespace {

inline std::size_t flat(int nx, int j, int i) {
  return static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(i);
}

// Homogeneous value outside the domain: advective inflow boundary.
inline double read_open(const double* c, int ny, int nx, int j, int i) {
  if (i < 0 || i >= nx || j < 0 || j >= ny) {
    return 0.0;
  }
  return c[flat(nx, j, i)];
}

// Clamped neighbour access: zero normal diffusive flux.
inline double read_wall(const double* c, int ny, int nx, int j, int i) {
  const int ic = std::clamp(i, 0, nx - 1);
  const int jc = std::clamp(j, 0, ny - 1);
  return c[flat(nx, jc, ic)];
}

inline void add_open(double* c, int ny, int nx, int j, int i, double value) {
  if (i < 0 || i >= nx || j < 0 || j >= ny) {
    return;
  }
  c[flat(nx, j, i)] += value;
}

inline void add_wall(double* c, int ny, int nx, int j, int i, double value) {
  const int ic = std::clamp(i, 0, nx - 1);
  const int jc = std::clamp(j, 0, ny - 1);
  c[flat(nx, jc, ic)] += value;
}

struct WindSplit {
  double wx_plus;
  double wx_minus;
  double wy_plus;
  double wy_minus;
};

inline WindSplit split(const TransportParams& p) {
  return WindSplit{std::max(p.wx, 0.0), std::min(p.wx, 0.0), std::max(p.wy, 0.0),
                   std::min(p.wy, 0.0)};
}

// Applies A(psi) to c and writes the result into out. When source is not null
// dt * source is added, giving the full explicit step of section 5.3.
void apply_step(const TransportParams& p, const double* c, const double* source,
                double* out) {
  const int ny = p.ny;
  const int nx = p.nx;
  const double dx = p.dx();
  const double dy = p.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const WindSplit w = split(p);

#pragma omp parallel for schedule(static)
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t k = flat(nx, j, i);
      const double centre = c[k];
      const double backward_x = (centre - read_open(c, ny, nx, j, i - 1)) / dx;
      const double forward_x = (read_open(c, ny, nx, j, i + 1) - centre) / dx;
      const double backward_y = (centre - read_open(c, ny, nx, j - 1, i)) / dy;
      const double forward_y = (read_open(c, ny, nx, j + 1, i) - centre) / dy;
      const double advection = -w.wx_plus * backward_x - w.wx_minus * forward_x -
                               w.wy_plus * backward_y - w.wy_minus * forward_y;
      const double laplacian =
          (read_wall(c, ny, nx, j, i + 1) - 2.0 * centre +
           read_wall(c, ny, nx, j, i - 1)) *
              inv_dx2 +
          (read_wall(c, ny, nx, j + 1, i) - 2.0 * centre +
           read_wall(c, ny, nx, j - 1, i)) *
              inv_dy2;
      double value = centre + p.dt * (advection + p.diffusivity * laplacian -
                                      p.decay * centre);
      if (source != nullptr) {
        value += p.dt * source[k];
      }
      out[k] = value;
    }
  }
}

// Applies A(psi)^T to y and accumulates the result into out.
//
// The transpose mirrors apply_step term by term, scattering each coefficient to
// the cell it reads in the forward operator. The scatter is coloured by
// j modulo 3 so that concurrently processed source rows never write to the same
// destination row; the result is bitwise independent of the thread count.
void apply_step_transpose(const TransportParams& p, const double* y, double* out) {
  const int ny = p.ny;
  const int nx = p.nx;
  const double dx = p.dx();
  const double dy = p.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const WindSplit w = split(p);
  const double dt = p.dt;
  const double dc = p.diffusivity;

  for (int colour = 0; colour < 3; ++colour) {
#pragma omp parallel for schedule(static)
    for (int j = colour; j < ny; j += 3) {
      for (int i = 0; i < nx; ++i) {
        const double yv = y[flat(nx, j, i)];
        if (yv == 0.0) {
          continue;
        }
        const double a = dt * yv;
        // Identity term.
        out[flat(nx, j, i)] += yv;
        // Advection.
        out[flat(nx, j, i)] += a * (-w.wx_plus / dx);
        add_open(out, ny, nx, j, i - 1, a * (w.wx_plus / dx));
        out[flat(nx, j, i)] += a * (w.wx_minus / dx);
        add_open(out, ny, nx, j, i + 1, a * (-w.wx_minus / dx));
        out[flat(nx, j, i)] += a * (-w.wy_plus / dy);
        add_open(out, ny, nx, j - 1, i, a * (w.wy_plus / dy));
        out[flat(nx, j, i)] += a * (w.wy_minus / dy);
        add_open(out, ny, nx, j + 1, i, a * (-w.wy_minus / dy));
        // Diffusion.
        const double cx = a * dc * inv_dx2;
        const double cy = a * dc * inv_dy2;
        add_wall(out, ny, nx, j, i + 1, cx);
        add_wall(out, ny, nx, j, i - 1, cx);
        add_wall(out, ny, nx, j + 1, i, cy);
        add_wall(out, ny, nx, j - 1, i, cy);
        out[flat(nx, j, i)] += -2.0 * (cx + cy);
        // Decay.
        out[flat(nx, j, i)] += a * (-p.decay);
      }
    }
  }
}

// Directional derivatives of A(psi) c with respect to the continuous
// parameters, evaluated on the upwind branch that is actually active.
struct StepSensitivities {
  double wind_x;
  double wind_y;
  double diffusivity;
  double decay;
};

StepSensitivities step_sensitivities(const TransportParams& p, const double* c,
                                     const double* cbar) {
  const int ny = p.ny;
  const int nx = p.nx;
  const double dx = p.dx();
  const double dy = p.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const bool wx_positive = p.wx > 0.0;
  const bool wy_positive = p.wy > 0.0;

  std::vector<double> row_wx(static_cast<std::size_t>(ny), 0.0);
  std::vector<double> row_wy(static_cast<std::size_t>(ny), 0.0);
  std::vector<double> row_dc(static_cast<std::size_t>(ny), 0.0);
  std::vector<double> row_lam(static_cast<std::size_t>(ny), 0.0);

#pragma omp parallel for schedule(static)
  for (int j = 0; j < ny; ++j) {
    double acc_wx = 0.0;
    double acc_wy = 0.0;
    double acc_dc = 0.0;
    double acc_lam = 0.0;
    for (int i = 0; i < nx; ++i) {
      const std::size_t k = flat(nx, j, i);
      const double centre = c[k];
      const double weight = cbar[k];
      if (weight == 0.0) {
        continue;
      }
      const double backward_x = (centre - read_open(c, ny, nx, j, i - 1)) / dx;
      const double forward_x = (read_open(c, ny, nx, j, i + 1) - centre) / dx;
      const double backward_y = (centre - read_open(c, ny, nx, j - 1, i)) / dy;
      const double forward_y = (read_open(c, ny, nx, j + 1, i) - centre) / dy;
      const double laplacian =
          (read_wall(c, ny, nx, j, i + 1) - 2.0 * centre +
           read_wall(c, ny, nx, j, i - 1)) *
              inv_dx2 +
          (read_wall(c, ny, nx, j + 1, i) - 2.0 * centre +
           read_wall(c, ny, nx, j - 1, i)) *
              inv_dy2;
      acc_wx += weight * (wx_positive ? -backward_x : -forward_x);
      acc_wy += weight * (wy_positive ? -backward_y : -forward_y);
      acc_dc += weight * laplacian;
      acc_lam += weight * (-centre);
    }
    row_wx[static_cast<std::size_t>(j)] = acc_wx;
    row_wy[static_cast<std::size_t>(j)] = acc_wy;
    row_dc[static_cast<std::size_t>(j)] = acc_dc;
    row_lam[static_cast<std::size_t>(j)] = acc_lam;
  }

  StepSensitivities total{0.0, 0.0, 0.0, 0.0};
  for (int j = 0; j < ny; ++j) {
    const auto index = static_cast<std::size_t>(j);
    total.wind_x += row_wx[index];
    total.wind_y += row_wy[index];
    total.diffusivity += row_dc[index];
    total.decay += row_lam[index];
  }
  total.wind_x *= p.dt;
  total.wind_y *= p.dt;
  total.diffusivity *= p.dt;
  total.decay *= p.dt;
  return total;
}

void scatter_observation_cotangent(const TransportParams& p,
                                   const SensorStencil& stencil,
                                   const double* cotangent_row, double* cbar) {
  const int nx = p.nx;
  for (int s = 0; s < stencil.size(); ++s) {
    const double value = cotangent_row[s];
    if (value == 0.0) {
      continue;
    }
    const int i0 = stencil.i0[static_cast<std::size_t>(s)];
    const int j0 = stencil.j0[static_cast<std::size_t>(s)];
    const double* w = &stencil.weights[static_cast<std::size_t>(s) * 4U];
    cbar[flat(nx, j0, i0)] += value * w[0];
    cbar[flat(nx, j0, i0 + 1)] += value * w[1];
    cbar[flat(nx, j0 + 1, i0)] += value * w[2];
    cbar[flat(nx, j0 + 1, i0 + 1)] += value * w[3];
  }
}

}  // namespace

double cfl_number(const TransportParams& p) {
  const double dx = p.dx();
  const double dy = p.dy();
  return p.dt * (std::abs(p.wx) / dx + std::abs(p.wy) / dy +
                 2.0 * p.diffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy)) +
                 p.decay);
}

void validate(const TransportParams& p) {
  if (p.nx < 3 || p.ny < 3) {
    throw std::invalid_argument("SmokeTransport grid must have at least 3 cells per axis");
  }
  if (p.n_steps < 1) {
    throw std::invalid_argument("SmokeTransport requires n_steps >= 1");
  }
  if (p.dt <= 0.0) {
    throw std::invalid_argument("SmokeTransport requires dt > 0");
  }
  if (p.diffusivity <= 0.0) {
    throw std::invalid_argument("SmokeTransport requires D_c > 0");
  }
  if (p.decay < 0.0) {
    throw std::invalid_argument("SmokeTransport requires lambda_c >= 0");
  }
  const double nu = cfl_number(p);
  if (nu > 1.0) {
    throw std::invalid_argument("SmokeTransport CFL violated: nu_c = " +
                                std::to_string(nu) + " > 1");
  }
}

SensorStencil build_stencil(const TransportParams& p, const double* positions,
                            int n_sensors) {
  if (n_sensors < 1) {
    throw std::invalid_argument("SmokeTransport requires at least one sensor");
  }
  SensorStencil stencil;
  stencil.i0.resize(static_cast<std::size_t>(n_sensors));
  stencil.j0.resize(static_cast<std::size_t>(n_sensors));
  stencil.weights.resize(static_cast<std::size_t>(n_sensors) * 4U);
  const double dx = p.dx();
  const double dy = p.dy();
  for (int s = 0; s < n_sensors; ++s) {
    const double sx = positions[static_cast<std::size_t>(s) * 2U];
    const double sy = positions[static_cast<std::size_t>(s) * 2U + 1U];
    const double gx = sx / dx - 0.5;
    const double gy = sy / dy - 0.5;
    const auto i0 = static_cast<int>(std::floor(gx));
    const auto j0 = static_cast<int>(std::floor(gy));
    if (i0 < 0 || i0 + 1 > p.nx - 1 || j0 < 0 || j0 + 1 > p.ny - 1) {
      throw std::invalid_argument(
          "SmokeTransport sensor outside the bilinear-safe domain window");
    }
    const double fx = gx - static_cast<double>(i0);
    const double fy = gy - static_cast<double>(j0);
    stencil.i0[static_cast<std::size_t>(s)] = i0;
    stencil.j0[static_cast<std::size_t>(s)] = j0;
    double* w = &stencil.weights[static_cast<std::size_t>(s) * 4U];
    w[0] = (1.0 - fy) * (1.0 - fx);
    w[1] = (1.0 - fy) * fx;
    w[2] = fy * (1.0 - fx);
    w[3] = fy * fx;
  }
  return stencil;
}

void forward_history(const TransportParams& p, const double* source, double* history) {
  validate(p);
  const std::size_t cells = p.cells();
  std::memset(history, 0, cells * sizeof(double));
  for (int n = 0; n < p.n_steps; ++n) {
    apply_step(p, history + static_cast<std::size_t>(n) * cells,
               source + static_cast<std::size_t>(n) * cells,
               history + static_cast<std::size_t>(n + 1) * cells);
  }
}

void observe(const TransportParams& p, const SensorStencil& stencil,
             const double* history, const double* bias, double* out) {
  const std::size_t cells = p.cells();
  const int n_sensors = stencil.size();
  const int nx = p.nx;
  for (int n = 1; n <= p.n_steps; ++n) {
    const double* c = history + static_cast<std::size_t>(n) * cells;
    for (int s = 0; s < n_sensors; ++s) {
      const int i0 = stencil.i0[static_cast<std::size_t>(s)];
      const int j0 = stencil.j0[static_cast<std::size_t>(s)];
      const double* w = &stencil.weights[static_cast<std::size_t>(s) * 4U];
      const double value = w[0] * c[flat(nx, j0, i0)] +
                           w[1] * c[flat(nx, j0, i0 + 1)] +
                           w[2] * c[flat(nx, j0 + 1, i0)] +
                           w[3] * c[flat(nx, j0 + 1, i0 + 1)];
      out[static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(n_sensors) +
          static_cast<std::size_t>(s)] =
          value + (bias == nullptr ? 0.0 : bias[static_cast<std::size_t>(s)]);
    }
  }
}

void extract_frames(const TransportParams& p, const double* history,
                    const int* frame_levels, int frame_count, double* out) {
  const std::size_t cells = p.cells();
  for (int f = 0; f < frame_count; ++f) {
    const int level = frame_levels[static_cast<std::size_t>(f)];
    if (level < 0 || level > p.n_steps) {
      throw std::invalid_argument("SmokeTransport frame level out of range");
    }
    std::memcpy(out + static_cast<std::size_t>(f) * cells,
                history + static_cast<std::size_t>(level) * cells,
                cells * sizeof(double));
  }
}

void vector_jacobian_product(const TransportParams& p, const SensorStencil& stencil,
                             const double* source, const double* cotangent,
                             double* source_bar, double* wind_bar,
                             double* diffusivity_bar, double* decay_bar) {
  validate(p);
  const std::size_t cells = p.cells();
  const int n_sensors = stencil.size();

  // Replay the forward pass; the endpoint is stateless between calls.
  std::vector<double> history(cells * static_cast<std::size_t>(p.n_steps + 1), 0.0);
  forward_history(p, source, history.data());

  std::memset(source_bar, 0, cells * static_cast<std::size_t>(p.n_steps) * sizeof(double));
  double grad_wx = 0.0;
  double grad_wy = 0.0;
  double grad_dc = 0.0;
  double grad_lam = 0.0;

  std::vector<double> cbar(cells, 0.0);
  std::vector<double> cbar_next(cells, 0.0);

  // Cotangent injected at the last observed level.
  scatter_observation_cotangent(
      p, stencil,
      cotangent + static_cast<std::size_t>(p.n_steps - 1) *
                      static_cast<std::size_t>(n_sensors),
      cbar.data());

  for (int n = p.n_steps - 1; n >= 0; --n) {
    const double* c_level = history.data() + static_cast<std::size_t>(n) * cells;
    double* s_bar = source_bar + static_cast<std::size_t>(n) * cells;
    for (std::size_t k = 0; k < cells; ++k) {
      s_bar[k] += p.dt * cbar[k];
    }
    const StepSensitivities sens = step_sensitivities(p, c_level, cbar.data());
    grad_wx += sens.wind_x;
    grad_wy += sens.wind_y;
    grad_dc += sens.diffusivity;
    grad_lam += sens.decay;

    std::fill(cbar_next.begin(), cbar_next.end(), 0.0);
    apply_step_transpose(p, cbar.data(), cbar_next.data());
    if (n >= 1) {
      scatter_observation_cotangent(
          p, stencil,
          cotangent + static_cast<std::size_t>(n - 1) *
                          static_cast<std::size_t>(n_sensors),
          cbar_next.data());
    }
    cbar.swap(cbar_next);
  }

  wind_bar[0] = grad_wx;
  wind_bar[1] = grad_wy;
  if (diffusivity_bar != nullptr) {
    *diffusivity_bar = grad_dc;
  }
  if (decay_bar != nullptr) {
    *decay_bar = grad_lam;
  }
}

}  // namespace climacare
