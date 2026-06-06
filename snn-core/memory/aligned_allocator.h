#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

namespace snn {
namespace core {
namespace memory {

/**
 * @brief Aligned Memory Allocator
 * 
 * Ensures memory allocations are aligned to 32-byte (AVX2) or 64-byte (AVX-512) 
 * boundaries. This is strictly required for maximum SIMD throughput and cache-line 
 * efficiency when doing billions of POPCOUNT operations.
 */
template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        size_type bytes = n * sizeof(T);
        void* ptr = nullptr;
        
#if defined(_MSC_VER) || defined(__MINGW32__)
        ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return false;
}

} // namespace memory
} // namespace core
} // namespace snn

