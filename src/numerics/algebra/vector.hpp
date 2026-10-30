#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <utility>

class Vector {
public:
    using Index = Eigen::Index;
    using NativeType = Eigen::VectorXd;

    Vector() = default;

    explicit Vector(Index size) : values_(size) {}

    Vector(std::initializer_list<double> values) : values_(static_cast<Index>(values.size())) {
        std::copy(values.begin(), values.end(), values_.data());
    }

    explicit Vector(NativeType values) : values_(std::move(values)) {}

    Index size() const noexcept {
        return values_.size();
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    void resize(Index size) {
        values_.resize(size);
    }

    double& operator[](Index index) {
        return values_[index];
    }

    const double& operator[](Index index) const {
        return values_[index];
    }

    void set_zero() {
        values_.setZero();
    }

    void set_constant(double value) {
        values_.setConstant(value);
    }

    double norm() const {
        return values_.norm();
    }

    double squared_norm() const {
        return values_.squaredNorm();
    }

    double infinity_norm() const {
        return values_.lpNorm<Eigen::Infinity>();
    }

    bool all_finite() const {
        return values_.allFinite();
    }

    NativeType& native() noexcept {
        return values_;
    }

    const NativeType& native() const noexcept {
        return values_;
    }

    Vector& operator+=(const Vector& other) {
        check_same_size(other);
        values_ += other.values_;
        return *this;
    }

    Vector& operator-=(const Vector& other) {
        check_same_size(other);
        values_ -= other.values_;
        return *this;
    }

    Vector& operator*=(double scalar) {
        values_ *= scalar;
        return *this;
    }

    Vector& operator/=(double scalar) {
        values_ /= scalar;
        return *this;
    }

    friend Vector operator+(Vector lhs, const Vector& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Vector operator-(Vector lhs, const Vector& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Vector operator-(Vector vector) {
        vector *= -1.0;
        return vector;
    }

    friend Vector operator*(Vector vector, double scalar) {
        vector *= scalar;
        return vector;
    }

    friend Vector operator*(double scalar, Vector vector) {
        vector *= scalar;
        return vector;
    }

    friend Vector operator/(Vector vector, double scalar) {
        vector /= scalar;
        return vector;
    }

private:
    void check_same_size(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector sizes do not match");
        }
    }

    NativeType values_;
};
