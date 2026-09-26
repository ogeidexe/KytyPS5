#ifndef EMULATOR_SRC_COMMON_FRAMESTATS_H_
#define EMULATOR_SRC_COMMON_FRAMESTATS_H_

#include "common/timer.h"

#include <atomic>
#include <cstdint>

// Counters behind the KYTY_FRAME_STATS=1 per-second report (see WindowContext::UpdateTitle),
// used to tell shader and pipeline compile stalls apart from other hitches.
namespace Common::FrameStats {

inline std::atomic<uint64_t> g_compile_us {0};
inline std::atomic<uint32_t> g_compile_count {0};

// Adds the lifetime of the scope to the compile counters.
class CompileScope {
public:
	CompileScope(): m_start(Timer::QueryPerformanceCounter()) {}
	~CompileScope() {
		const auto elapsed = Timer::QueryPerformanceCounter() - m_start;
		g_compile_us.fetch_add(elapsed * 1000000 / Timer::QueryPerformanceFrequency(),
		                       std::memory_order_relaxed);
		g_compile_count.fetch_add(1, std::memory_order_relaxed);
	}

	CompileScope(const CompileScope&)            = delete;
	CompileScope& operator=(const CompileScope&) = delete;

private:
	uint64_t m_start;
};

} // namespace Common::FrameStats

#endif // EMULATOR_SRC_COMMON_FRAMESTATS_H_
