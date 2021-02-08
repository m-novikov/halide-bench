#include <cstdio>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Halide.h"

using namespace std;
using namespace Halide;

namespace py = pybind11;

struct GaussianBlur {
public:
     GaussianBlur(const float sigma) {
         _sigma = sigma;
        const int boxSize = 2 * ceil(3 * sigma) + 1;
        const int radius = (boxSize - 1) / 2;
        vector<float> k(radius + 1);
        float ksum = 0.0f;
        for (int i = 0;  i <= radius; ++i) {
            const float ki = exp(-i * i / (2.0f * sigma * sigma)) / (sqrtf(2 * M_PI) * sigma);
            k[i] = ki;
            if (i == 0) {
                ksum += ki;
            } else {
                ksum += 2 * ki;
            }
        }
        Buffer<float> kernel(radius + 1);
        for (int i = 0; i < k.size(); ++i) {
            kernel(i) = k[i] / ksum;
        }
        _kernel = kernel;
        _in = ImageParam(Float(32), 2);
        Var x, y;
        Func extended("extended");
        Func blur_x("blur_x");
        Func blur_y("blur_y");
        extended = Halide::BoundaryConditions::mirror_interior(_in);
        RDom r(1, radius);
        blur_y(x, y) = kernel(0) * extended(x, y) + sum(kernel(r) * (extended(x, y - r) + extended(x, y + r)));
        blur_x(x, y) = kernel(0) * blur_y(x, y) + sum(kernel(r) * (blur_y(x - r, y) + blur_y(x + r, y)));
        blur_x.compute_root().vectorize(x, 16);
        blur_y.compute_at(blur_x, y).vectorize(x, 16);
        blur_x.compile_jit();
        _blur = blur_x;
     }

    py::array_t<float> compute(py::array_t<float, py::array::c_style> arr) {
        py::buffer_info arr_buf = arr.request();
        Buffer<float> input((float*)arr_buf.ptr, arr.shape()[1], arr.shape()[0]); 
        _in.set(input);
        auto result = py::array_t<float>(arr.size());
        auto result_info = result.request();
        Halide::Buffer<float> output((float*)result_info.ptr, arr.shape()[1], arr.shape()[0]);
        _blur.realize(output);
        result.resize({arr.shape()[0], arr.shape()[1]});
        return result;
    }
private:
     float _sigma;
     Buffer<float> _kernel;
     ImageParam _in;
     Func _blur;
     OutputImageParam _out;
};


struct GaussianBlur3D {
public:
     GaussianBlur3D(const float sigma) {
         _sigma = sigma;
        const int boxSize = 2 * ceil(3 * sigma) + 1;
        const int radius = (boxSize - 1) / 2;
        vector<float> k(radius + 1);
        float ksum = 0.0f;
        for (int i = 0;  i <= radius; ++i) {
            const float ki = exp(-i * i / (2.0f * sigma * sigma)) / (sqrtf(2 * M_PI) * sigma);
            k[i] = ki;
            if (i == 0) {
                ksum += ki;
            } else {
                ksum += 2 * ki;
            }
        }
        Buffer<float> kernel(radius + 1);
        for (int i = 0; i < k.size(); ++i) {
            kernel(i) = k[i] / ksum;
        }
        _kernel = kernel;
        _in = ImageParam(Float(32), 3);
        Var x, y, z;
        Func extended("extended");
        Func blur_x("blur_x");
        Func blur_y("blur_y");
        Func blur_z("blur_z");
        extended = Halide::BoundaryConditions::mirror_interior(_in);
        RDom r(1, radius);
        blur_z(x, y, z) = kernel(0) * extended(x, y, z) + sum(kernel(r) * (extended(x, y, z - r) + extended(x, y, z + r)));
        blur_y(x, y, z) = kernel(0) * blur_z(x, y, z) + sum(kernel(r) * (blur_z(x, y - r, z) + blur_z(x, y + r, z)));
        blur_x(x, y, z) = kernel(0) * blur_y(x, y, z) + sum(kernel(r) * (blur_y(x - r, y, z) + blur_y(x + r, y, z)));
        blur_x.compute_root().vectorize(x, 16);
        blur_y.compute_at(blur_x, y).vectorize(x, 16);
        blur_z.compute_at(blur_x, z).vectorize(x, 16);
        blur_x.compile_jit();
        _blur = blur_x;
     }

    py::array_t<float> compute(py::array_t<float, py::array::c_style> arr) {
        py::buffer_info arr_buf = arr.request();
        Buffer<float> input((float*)arr_buf.ptr, arr.shape()[2], arr.shape()[1], arr.shape()[0]);
        _in.set(input);
        auto result = py::array_t<float>(arr.size());
        auto result_info = result.request();
        Halide::Buffer<float> output((float*)result_info.ptr, arr.shape()[2], arr.shape()[1], arr.shape()[0]);
        _blur.realize(output);
        result.resize({arr.shape()[0], arr.shape()[1], arr.shape()[2]});
        return result;
    }
private:
     float _sigma;
     Buffer<float> _kernel;
     ImageParam _in;
     Func _blur;
     OutputImageParam _out;
};


PYBIND11_MODULE(filterbench, m) {
    py::class_<GaussianBlur>(m, "GaussianBlur")
        .def(py::init<const float>())
        .def("compute", &GaussianBlur::compute);
    py::class_<GaussianBlur3D>(m, "GaussianBlur3D")
        .def(py::init<const float>())
        .def("compute", &GaussianBlur3D::compute);
}
