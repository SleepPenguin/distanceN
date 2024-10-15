#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "helper_cuda.h"
#include "ticktock.h"

using Dvector = thrust::device_vector<double>;
using Hvector = thrust::host_vector<double>;

template<typename T>
std::ostream &operator<<(std::ostream &os, const thrust::host_vector<T> &h_vec) {
    os << "[";
    for (size_t i = 0; i < h_vec.size(); ++i) {
        os << h_vec[i];
        if (i != h_vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// cal core
struct distance {
    int N{};
    double *pD_vec;

    distance(const int n, Dvector &d_vec) : N(n), pD_vec(thrust::raw_pointer_cast(d_vec.data())) {
    }

    __device__ double operator()(const int &idx) const {
        // idx: 1 -> N*N
        const int i = idx / N;
        const int j = idx % N;
        return ((pD_vec[i] - pD_vec[j]) * (pD_vec[i] - pD_vec[j]) +
                (pD_vec[i + N] - pD_vec[j + N]) * (pD_vec[i + N] - pD_vec[j + N]) +
                (pD_vec[i + 2 * N] - pD_vec[j + 2 * N]) * (pD_vec[i + 2 * N] - pD_vec[j + 2 * N]));
    }
};

void save(Hvector const &v, const std::string &filename) {
    if (std::ofstream out(filename, std::ios::binary); out.is_open()) {
        const size_t size = v.size();
        out.write(reinterpret_cast<const char *>(thrust::raw_pointer_cast(v.data())), size * sizeof(double));
        out.close();
    } else {
        std::cerr << "Can't open file " << filename << "\n";
    }
}

int main() {
    // N 在 4e4往上会出现int溢出的问题还需要处理
    constexpr int N = 1000;

    // Generate 3N random numbers on device.
    thrust::default_random_engine rng(1332);
    thrust::uniform_real_distribution<double> distribute(-50.0, 50.0);
    Hvector h_vec(3 * N);
    thrust::generate(h_vec.begin(), h_vec.end(), [&] { return distribute(rng); });


    // x0,x1,x2...,y0,y1...,z0,z1...,zN  3*N
    Dvector d_vec = h_vec;

    Dvector d_dis2(N * N);

    TICK(distanceCore);

    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(N * N), d_dis2.begin(), distance(N, d_vec));

    checkCudaErrors(cudaDeviceSynchronize());

    TOCK(distanceCore);

    Hvector h_dis2 = d_dis2; // 考虑这个拷贝耗时会大大增加，瓶颈主要是数据IO

    // std::cout << h_vec << std::endl;
    // std::cout << h_dis2 << std::endl;

    // check errors
    if (N <= 1000) {
        save(h_vec, "input.dat");
        save(h_dis2, "output.dat");
        TICK(errorCheck);
        for (int id = 0; id < h_dis2.size(); id++) {
            const int i = id / N;
            const int j = id % N;
            const double h_res = (h_vec[i] - h_vec[j]) * (h_vec[i] - h_vec[j]) +
                                 (h_vec[i + N] - h_vec[j + N]) * (h_vec[i + N] - h_vec[j + N]) +
                                 (h_vec[i + 2 * N] - h_vec[j + 2 * N]) * (h_vec[i + 2 * N] - h_vec[j + 2 * N]);
            if ((h_res - h_dis2[id]) > 1e-5)
                printf("error id %d\n, cpu res %.4f, gpu res %.4f\n", id, h_res, h_dis2[id]);
        }
        TOCK(errorCheck);
    } else {
        std::cout << "N is too large, skip save and check\n";
    }
}
