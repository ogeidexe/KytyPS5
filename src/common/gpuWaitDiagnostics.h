#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_GPUWAITDIAGNOSTICS_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_GPUWAITDIAGNOSTICS_H_

// Opt-in (KYTY_GPU_WAIT_DIAGNOSTICS=1) trail of GPU queue events, dumped when a host wait for the
// GPU takes unusually long, so a stuck wait names the submission it is stuck behind. Recording is
// a no-op unless enabled; enabling it never changes synchronization, only reports on it.

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

namespace Common::GpuWaitDiagnostics {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_WAIT_DIAGNOSTICS");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

struct Event {
	uint64_t    time_us = 0;
	uint64_t    thread  = 0;
	const char* what    = nullptr; // string literal
	uint64_t    a = 0, b = 0, c = 0, d = 0;
};

struct Trail {
	std::mutex              mutex;
	std::array<Event, 512>  events {};
	uint64_t                next = 0;
};

inline Trail& GetTrail() {
	static Trail trail;
	return trail;
}

inline uint64_t NowUs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline void Note(const char* what, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0,
                 uint64_t d = 0) {
	if (!Enabled()) {
		return;
	}
	auto&           trail = GetTrail();
	std::lock_guard lock(trail.mutex);
	trail.events[trail.next++ % trail.events.size()] = {
	    NowUs(), std::hash<std::thread::id> {}(std::this_thread::get_id()), what, a, b, c, d};
}

inline void Dump(const char* header) {
	auto&           trail = GetTrail();
	std::lock_guard lock(trail.mutex);
	const auto      now   = NowUs();
	const auto      count = std::min<uint64_t>(trail.next, trail.events.size());
	std::printf("[gpu-wait] %s; last %" PRIu64 " queue events (newest last):\n", header, count);
	for (uint64_t i = trail.next - count; i < trail.next; i++) {
		const auto& e = trail.events[i % trail.events.size()];
		std::printf("[gpu-wait]   -%9.3f ms  thr=%04" PRIx64 " %-18s a=%" PRIu64 " b=%" PRIu64
		            " c=%" PRIu64 " d=%" PRIu64 "\n",
		            static_cast<double>(now - e.time_us) / 1000.0, e.thread & 0xffffu, e.what, e.a,
		            e.b, e.c, e.d);
	}
	std::fflush(stdout);
}

} // namespace Common::GpuWaitDiagnostics

#endif // EMULATOR_INCLUDE_EMULATOR_COMMON_GPUWAITDIAGNOSTICS_H_
