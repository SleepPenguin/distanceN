#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "helper_cuda.h"
#include "ticktock.h"

using Dvector = thrust::device_vector<double>;
using Hvector = thrust::host_vector<double>;

template <typename T>
std::ostream &operator<<(std::ostream &os, const thrust::host_vector<T> &h_vec)
{
    os << "[";
    for (size_t i = 0; i < h_vec.size(); ++i)
    {
        os << h_vec[i];
        if (i != h_vec.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// cal core
struct distance
{
    int N{};
    double *pD_vec;

    distance(int n, Dvector &d_vec) : N(n), pD_vec(thrust::raw_pointer_cast(d_vec.data())) {}

    __device__ double operator()(const int &idx)
    {
        // idx: 1 -> N*N
        int i = idx / N;
        int j = idx % N;
        return ((pD_vec[i] - pD_vec[j]) * (pD_vec[i] - pD_vec[j]) +
                (pD_vec[i + N] - pD_vec[j + N]) * (pD_vec[i + N] - pD_vec[j + N]) +
                (pD_vec[i + 2 * N] - pD_vec[j + 2 * N]) * (pD_vec[i + 2 * N] - pD_vec[j + 2 * N]));
    }
};

void save(Hvector const &v, std::string filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (out.is_open())
    {
        size_t size = v.size();
        out.write(reinterpret_cast<const char *>(thrust::raw_pointer_cast(v.data())), size * sizeof(double));
        out.close();
    }
    else
    {
        std::cerr << "Can't open file " << filename << "\n";
    }
}

int main()
{
    int N = 1000;

    // Generate 3N random numbers on device.
    thrust::default_random_engine rng(1332);
    thrust::uniform_real_distribution<double> distribute(-50.0, 50.0);
    Hvector h_vec(3 * N);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]
                     { return distribute(rng); });

    save(h_vec, "input.dat");

    // x0,x1,x2...,y0,y1...,z0,z1...,zN  3*N
    Dvector d_vec = h_vec;

    Dvector d_dis2(N * N);

    TICK(distanceCore);

    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(N * N), d_dis2.begin(), distance(N, d_vec));

    checkCudaErrors(cudaDeviceSynchronize()); // spend more time than core
    TOCK(distanceCore);

    Hvector h_dis2 = d_dis2;
    save(h_dis2, "output.dat");

    // std::cout << h_vec << std::endl;
    // std::cout << h_dis2 << std::endl;

    // check errors
    TICK(errorCheck);
    for (int id = 0; id < h_dis2.size(); id++)
    {
        int i = id / N;
        int j = id % N;
        double h_res = (h_vec[i] - h_vec[j]) * (h_vec[i] - h_vec[j]) +
                       (h_vec[i + N] - h_vec[j + N]) * (h_vec[i + N] - h_vec[j + N]) +
                       (h_vec[i + 2 * N] - h_vec[j + 2 * N]) * (h_vec[i + 2 * N] - h_vec[j + 2 * N]);
        if ((h_res - h_dis2[id]) > 1e-5)
            printf("error id %d\n, cpu res %.4f, gpu res %.4f\n", id, h_res, h_dis2[id]);
    }
    TOCK(errorCheck);
}