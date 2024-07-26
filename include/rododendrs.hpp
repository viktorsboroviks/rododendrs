#include <mutex>
#include <random>

namespace rododendrs {

class Random {
    // thread-safe implementation of rnd01() borrowed from
    // https://github.com/Arash-codedev/openGA/blob/master/README.md
    // assuming those people knew what they were doing
private:
    std::mutex mtx_rand;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif_dist;

public:
    Random()
    {
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();
        std::seed_seq ss{uint32_t(timeSeed & 0xffffffff),
                         uint32_t(timeSeed >> 32)};
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(0, 1);
    }

    Random(const Random &other)
    {
        // construct a new object on copy
        (void)other;
        Random();
    }

    double rnd01(void)
    {
        // prevent data race between threads
        std::lock_guard<std::mutex> lock(mtx_rand);
        return unif_dist(rng);
    }
};

}  // namespace rododendrs
