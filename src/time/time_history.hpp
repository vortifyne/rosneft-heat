#pragma once

#include "time/time_snapshot.hpp"

#include <cstddef>
#include <deque>
#include <utility>

class TimeHistory {
public:
    explicit TimeHistory(std::size_t capacity = 1) : capacity_(capacity) {}

    void set_capacity(std::size_t capacity) {
        capacity_ = capacity;

        while (snapshots_.size() > capacity_) {
            snapshots_.pop_back();
        }
    }

    const TimeSnapshot& current() const {
        return snapshots_.front();
    }
    const TimeSnapshot& previous(std::size_t offset = 1) const {
        return snapshots_[offset];
    }
    std::size_t size() const {
        return snapshots_.size();
    }
    bool empty() const {
        return snapshots_.empty();
    }

    void clear() {
        snapshots_.clear();
    }

    void accept(TimeSnapshot&& snapshot) {
        snapshots_.push_front(std::move(snapshot));

        while (snapshots_.size() > capacity_) {
            snapshots_.pop_back();
        }
    }

private:
    std::size_t capacity_;
    std::deque<TimeSnapshot> snapshots_;
};
