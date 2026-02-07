#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstring>

// ============================================================================
// 1. ReLU: Sets negatives to 0
// ============================================================================
void naive_relu(float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// ============================================================================
// 2. Dot Product: Sum of A[i] * B[i]
// ============================================================================
float naive_dot(const float* A, const float* B, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += A[i] * B[i];
    }
    return sum;
}

// ============================================================================
// 3. Naive Matrix Multiplication (O(N^3))
// ============================================================================
void naive_matrix_mul(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// 4. Linked List Traversal (Memory Latency Test)
//    Random pointer chasing breaks CPU prefetcher
// ============================================================================
struct Node {
    Node* next;
    int value;
    char padding[56]; // Pad to 64 bytes (cache line size)
};

int naive_linked_list_sum(Node* head) {
    int sum = 0;
    Node* current = head;
    while (current != nullptr) {
        sum += current->value;
        current = current->next;
    }
    return sum;
}

// Helper to create a randomized linked list
Node* create_random_linked_list(size_t n, std::vector<Node>& storage) {
    storage.resize(n);
    
    // Initialize values
    for (size_t i = 0; i < n; ++i) {
        storage[i].value = static_cast<int>(i + 1);
        storage[i].next = nullptr;
    }
    
    // Create random order using Fisher-Yates shuffle of indices
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    
    // Link nodes in shuffled order
    for (size_t i = 0; i < n - 1; ++i) {
        storage[indices[i]].next = &storage[indices[i + 1]];
    }
    
    return &storage[indices[0]];
}

// ============================================================================
// 5. QuickSort (Branch Predictor Test)
//    Unpredictable comparisons stress branch prediction
// ============================================================================
int partition(int* arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[hi]);
    return i + 1;
}

void naive_quicksort(int* arr, int lo, int hi) {
    if (lo < hi) {
        int p = partition(arr, lo, hi);
        naive_quicksort(arr, lo, p - 1);
        naive_quicksort(arr, p + 1, hi);
    }
}

// ============================================================================
// 6. SHA-256 (Integer ALU / Crypto Test)
//    Heavy integer operations, bit manipulation
// ============================================================================

// SHA-256 constants
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

void naive_sha256_block(const uint8_t block[64], uint32_t hash[8]) {
    uint32_t W[64];
    
    // Prepare message schedule
    for (int i = 0; i < 16; ++i) {
        W[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | block[i*4+3];
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    }
    
    // Initialize working variables
    uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    
    // 64 rounds
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        uint32_t T2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    
    // Update hash
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

void naive_sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    // Initial hash values
    uint32_t hash[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process complete blocks
    size_t num_blocks = len / 64;
    for (size_t i = 0; i < num_blocks; ++i) {
        naive_sha256_block(data + i * 64, hash);
    }
    
    // Handle padding (simplified - assumes len < 55 for final block)
    uint8_t final_block[64] = {0};
    size_t remaining = len % 64;
    memcpy(final_block, data + num_blocks * 64, remaining);
    final_block[remaining] = 0x80;
    
    if (remaining >= 56) {
        naive_sha256_block(final_block, hash);
        memset(final_block, 0, 64);
    }
    
    // Append length in bits
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; ++i) {
        final_block[63 - i] = (bit_len >> (i * 8)) & 0xFF;
    }
    naive_sha256_block(final_block, hash);
    
    // Output hash
    for (int i = 0; i < 8; ++i) {
        out[i*4]   = (hash[i] >> 24) & 0xFF;
        out[i*4+1] = (hash[i] >> 16) & 0xFF;
        out[i*4+2] = (hash[i] >> 8) & 0xFF;
        out[i*4+3] = hash[i] & 0xFF;
    }
}

// ============================================================================
// 7. Fibonacci Recursive (Stack Overhead Test)
//    Deep recursion stresses call stack and dependency chains
// ============================================================================
uint64_t naive_fibonacci(int n) {
    if (n <= 1) return n;
    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2);
}

// ============================================================================
// 8. Memcpy (DRAM Bandwidth Test)
//    Raw memory bandwidth measurement
// ============================================================================
void naive_memcpy(void* dst, const void* src, size_t n) {
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}
