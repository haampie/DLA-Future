#include <iostream>
#include <cassert>
#include <memory>
#include <bitset>

#include <blas.hh>

#include "miniapp/timer.h"

template <class T>
struct Matrix {
  Matrix(int m, int n) : m_(m), n_(n), ld_(m), mem_(std::make_unique<T[]>(m * n)) {}

  T& operator()(int r, int c) {
    auto linear_index = c * m_ + r;
    assert(r >= 0 && r < m_);
    assert(c >= 0 && c < n_);

    return mem_[linear_index];
  }

  const T& operator()(int r, int c) const {
    auto linear_index = c * m_ + r;
    assert(r >= 0 && r < m_);
    assert(c >= 0 && c < n_);

    return mem_[linear_index];
  }

  int m_, n_, ld_;
  std::unique_ptr<T[]> mem_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
  for (auto r = 0; r < matrix.m_; ++r) {
    for (auto c = 0; c < matrix.n_; ++c)
      os << matrix(r, c) << ", ";
    os << std::endl;
  }
  return os;
}

template <class T, class Func>
void set(Matrix<T>& matrix, Func func) {
  for (auto c = 0; c < matrix.n_; ++c)
    for (auto r = 0; r < matrix.m_; ++r)
      matrix(r, c) = func(r, c);
}

using T = double;
const std::size_t N = 1024;     // number of time to run the test
const int m = 256;
const int n = 256;
const int k = 256;

auto gemm_loop(const std::size_t N, const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
  blas::Op op_a = blas::Op::NoTrans;
  blas::Op op_b = blas::Op::ConjTrans;
  T alpha = 1.0;
  T beta = 1.0;

  assert(a.n_ == b.m_);
  blas::gemm(blas::Layout::ColMajor, op_a, op_b, a.m_, b.n_, a.n_, alpha, a.mem_.get(), a.ld_, b.mem_.get(), b.ld_, beta, c.mem_.get(), c.ld_);

  dlaf::common::timer<> timeit;
  for (int i = 0; i < N; ++i)
    blas::gemm(blas::Layout::ColMajor, op_a, op_b, a.m_, b.n_, a.n_, alpha, a.mem_.get(), a.ld_, b.mem_.get(), b.ld_, beta, c.mem_.get(), c.ld_);
  auto elapsed_time = timeit.elapsed();

  return elapsed_time;
}

void test_trivial() {
  Matrix<T> a(m, k);
  Matrix<T> b(k, n);
  Matrix<T> c(m, n);

  set(a, [](int i, int j) { return i == j ? 1 : 0; });
  set(b, [](int i, int j) { return i == j ? 1 : 0; });
  set(c, [](int i, int j) { return 0; });

  auto time = gemm_loop(N, a, b, c);

  std::cout << "trivial " << time << std::endl;
}

void test_cholesky_case() {
  Matrix<T> a(m, k);
  Matrix<T> b(k, n);
  Matrix<T> c(m, n);

  set(a, [](int i, int j) { i+=768; return 1./std::exp2(i - j); });
  set(b, [](int i, int j) { i+=256; return 1./std::exp2(i - j); });
  set(c, [](int i, int j) { i+=768; j+=256; return 1./std::exp2(i - j); ; });

  auto time = gemm_loop(N, a, b, c);

  std::cout << "cholesky " << time << std::endl;
}

void test_template(std::string test_name, T value) {
  Matrix<T> a(m, k);
  Matrix<T> b(k, n);
  Matrix<T> c(m, n);

  set(a, [value](int i, int j) { return value; });
  set(b, [value](int i, int j) { return value; });
  set(c, [](int i, int j) { return 0; });

  auto time = gemm_loop(N, a, b, c);

  std::cout << test_name << " " << time << std::endl;

  T expected = (1 + N) * a.m_ * (value * value);

  if (c(0, 0) != expected)
    std::cout << "ERROR" << std::endl;
  else
    std::cout << std::bitset<64>(*reinterpret_cast<uint64_t*>(&c(0, 0))) << std::endl;
}

int main() {
  test_trivial();
  test_cholesky_case();
  test_template("basic", 1);
  test_template("bad", 1./std::exp2(535));

  return 0;
}
