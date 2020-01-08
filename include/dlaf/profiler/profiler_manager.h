#pragma once

#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <map>
#include <deque>

namespace dlaf {
namespace profiler {
namespace details {

  struct task_data_t {
    using clock_t = std::chrono::steady_clock;

    std::string name_;
    std::string group_;

    struct {
      std::thread::id id_;
      std::size_t time_;
    } start_data_, end_data_;

    void enter(const std::string& name, const std::string& group) {
      name_ = name;
      group_ = group;
      start_data_ = { std::this_thread::get_id(), get_time() };
    }

    void leave() {
      end_data_ = { std::this_thread::get_id(), get_time() };
    }

    std::size_t get_time() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock_t::now().time_since_epoch()
          ).count();
    };

    friend std::ostream& operator<<(std::ostream& os, const task_data_t& task_data) {
      os << task_data.name_ << ", ";
      os << task_data.group_ << ", ";
      os << task_data.start_data_.id_ << ", ";
      os << task_data.start_data_.time_ << ", ";
      os << task_data.end_data_.id_ << ", ";
      os << task_data.end_data_.time_;
      return os;
    }
  };

}

struct Manager {
  ~Manager() {
    std::ofstream profiler_report("report.csv");  // TODO adapt for multiple nodes
    for (const auto& recorder : recorders_)
      for (const auto& task : recorder.second.tasks_)
        profiler_report << task << std::endl;
  }

  struct LocalRecorder {
    std::deque<details::task_data_t> tasks_;
  };

  static Manager& get_global_profiler() {
    static Manager global_;
    return global_;
  }

  void add(const details::task_data_t& task_data) {
    thread_local std::thread::id tid = std::this_thread::get_id();
    recorders_[tid].tasks_.emplace_back(task_data);
  }

  std::map<std::thread::id, LocalRecorder> recorders_;
};

struct SectionScoped {
  SectionScoped(const std::string& task_name, const std::string& task_group) {
    data_.enter(task_name, task_group);
  }

  ~SectionScoped() {
    data_.leave();

    Manager::get_global_profiler().add(data_);
  }

  details::task_data_t data_;
};

namespace util {

template <class Func>
auto unwrap(std::string name, std::string group, Func&& target_function) {
  return [name, group, function=std::forward<Func>(target_function)](auto&&... args) -> auto {
    SectionScoped _(name, group);
    return function(std::forward<decltype(args)>(args)...);
  };
}

}

}
}
