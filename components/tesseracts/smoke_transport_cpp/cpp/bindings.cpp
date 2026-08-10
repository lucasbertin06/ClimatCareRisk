// Copyright 2026 ClimaCare-Risk contributors. SPDX-License-Identifier: Apache-2.0
//
// pybind11 glue for the SmokeTransport kernel. Tesseract requires a Python
// entry point, but the numerics and the discrete adjoint stay in C++20/OpenMP;
// this file only converts contiguous float64 buffers.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "smoke_kernel.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

climacare::TransportParams make_params(const Array& source, double wind_x,
                                       double wind_y, double diffusivity,
                                       double decay, double dt) {
  if (source.ndim() != 3) {
    throw std::invalid_argument("source must have shape (n_steps, ny, nx)");
  }
  climacare::TransportParams params;
  params.n_steps = static_cast<int>(source.shape(0));
  params.ny = static_cast<int>(source.shape(1));
  params.nx = static_cast<int>(source.shape(2));
  params.wx = wind_x;
  params.wy = wind_y;
  params.diffusivity = diffusivity;
  params.decay = decay;
  params.dt = dt;
  return params;
}

Array check_sensors(const Array& sensors) {
  if (sensors.ndim() != 2 || sensors.shape(1) != 2) {
    throw std::invalid_argument("sensor_positions must have shape (S, 2)");
  }
  return sensors;
}

}  // namespace

PYBIND11_MODULE(smoke_kernel_cpp, module) {
  module.doc() =
      "C++20/OpenMP explicit smoke transport solver with a discrete adjoint "
      "(ClimaCare-Risk C0, specification sections 5 and 6).";

  module.def(
      "cfl_number",
      [](const Array& source, double wind_x, double wind_y, double diffusivity,
         double decay, double dt) {
        return climacare::cfl_number(
            make_params(source, wind_x, wind_y, diffusivity, decay, dt));
      },
      py::arg("source"), py::arg("wind_x"), py::arg("wind_y"),
      py::arg("diffusivity"), py::arg("decay"), py::arg("dt"),
      "Return the sufficient stability budget nu_c for the given inputs.");

  module.def(
      "forward",
      [](const Array& source, const Array& sensors, const Array& bias, double wind_x,
         double wind_y, double diffusivity, double decay, double dt,
         const IntArray& frame_levels) {
        const auto params =
            make_params(source, wind_x, wind_y, diffusivity, decay, dt);
        climacare::validate(params);
        check_sensors(sensors);
        const auto n_sensors = static_cast<int>(sensors.shape(0));
        if (bias.ndim() != 1 || bias.shape(0) != n_sensors) {
          throw std::invalid_argument("sensor_bias must have shape (S,)");
        }
        const auto stencil =
            climacare::build_stencil(params, sensors.data(), n_sensors);

        std::vector<double> history(
            params.cells() * static_cast<std::size_t>(params.n_steps + 1), 0.0);
        Array observations({params.n_steps, n_sensors});
        const auto frame_count = static_cast<int>(frame_levels.shape(0));
        Array frames({frame_count, params.ny, params.nx});

        {
          py::gil_scoped_release release;
          climacare::forward_history(params, source.data(), history.data());
          climacare::observe(params, stencil, history.data(), bias.data(),
                             observations.mutable_data());
          climacare::extract_frames(params, history.data(), frame_levels.data(),
                                    frame_count, frames.mutable_data());
        }
        return py::make_tuple(observations, frames);
      },
      py::arg("source"), py::arg("sensor_positions"), py::arg("sensor_bias"),
      py::arg("wind_x"), py::arg("wind_y"), py::arg("diffusivity"), py::arg("decay"),
      py::arg("dt"), py::arg("frame_levels"),
      "Run the explicit solver and return (sensor_observations, frames).");

  module.def(
      "vector_jacobian_product",
      [](const Array& source, const Array& sensors, double wind_x, double wind_y,
         double diffusivity, double decay, double dt, const Array& cotangent) {
        const auto params =
            make_params(source, wind_x, wind_y, diffusivity, decay, dt);
        climacare::validate(params);
        check_sensors(sensors);
        const auto n_sensors = static_cast<int>(sensors.shape(0));
        if (cotangent.ndim() != 2 || cotangent.shape(0) != params.n_steps ||
            cotangent.shape(1) != n_sensors) {
          throw std::invalid_argument(
              "sensor cotangent must have shape (n_steps, S)");
        }
        const auto stencil =
            climacare::build_stencil(params, sensors.data(), n_sensors);

        Array source_bar({params.n_steps, params.ny, params.nx});
        Array wind_bar(2);
        double diffusivity_bar = 0.0;
        double decay_bar = 0.0;
        {
          py::gil_scoped_release release;
          climacare::vector_jacobian_product(
              params, stencil, source.data(), cotangent.data(),
              source_bar.mutable_data(), wind_bar.mutable_data(), &diffusivity_bar,
              &decay_bar);
        }
        return py::make_tuple(source_bar, wind_bar, diffusivity_bar, decay_bar);
      },
      py::arg("source"), py::arg("sensor_positions"), py::arg("wind_x"),
      py::arg("wind_y"), py::arg("diffusivity"), py::arg("decay"), py::arg("dt"),
      py::arg("cotangent"),
      "Return (source_bar, wind_bar, diffusivity_bar, decay_bar) from the "
      "discrete adjoint of the explicit scheme.");
}
