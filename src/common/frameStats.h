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
// Deepest audio output queue seen since the last report, and underruns in that window.
inline std::atomic<uint32_t> g_audio_queue_max_ms {0};
inline std::atomic<uint32_t> g_audio_underruns {0};
// Video latency: longest wait of a guest submission before the GPU thread started it, and the
// longest and summed time from a flip request to its presentation.
inline std::atomic<uint32_t> g_submit_wait_max_us {0};
inline std::atomic<uint32_t> g_flip_latency_max_us {0};
inline std::atomic<uint64_t> g_flip_latency_sum_us {0};
inline std::atomic<uint32_t> g_flip_count {0};
inline std::atomic<uint32_t> g_flip_pending_max {0};
// Synchronous GPU-to-CPU buffer readbacks and the time spent waiting on them.
inline std::atomic<uint32_t> g_readback_count {0};
inline std::atomic<uint64_t> g_readback_wait_us {0};
inline std::atomic<uint64_t> g_readback_bytes {0};
// Indirect draws: total, whose arguments were GPU-written (a CPU read stalls), that turned out
// to draw nothing, served from the speculation cache, and time stalled reading arguments.
inline std::atomic<uint32_t> g_indirect_draws {0};
inline std::atomic<uint32_t> g_indirect_gpu_written {0};
inline std::atomic<uint32_t> g_indirect_zero {0};
inline std::atomic<uint32_t> g_indirect_speculated {0};
inline std::atomic<uint64_t> g_indirect_stall_us {0};

inline void StoreMax(std::atomic<uint32_t>& target, uint32_t value) {
	auto seen = target.load(std::memory_order_relaxed);
	while (value > seen && !target.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}
inline std::atomic<uint32_t> g_image_creates {0};
inline std::atomic<uint32_t> g_image_deletes {0};

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
