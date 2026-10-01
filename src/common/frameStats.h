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
// Guest flips requested since boot (never reset): the guest frame number input recording uses.
inline std::atomic<uint64_t> g_guest_flips {0};
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
// Occlusion queries: counter dumps (ZPASS_DONE), ZPASS predications, how many of those skipped
// their packets and how many found the results not ready yet.
inline std::atomic<uint32_t> g_occlusion_dumps {0};
inline std::atomic<uint32_t> g_zpass_predications {0};
inline std::atomic<uint32_t> g_zpass_predication_skips {0};
inline std::atomic<uint32_t> g_zpass_predication_not_ready {0};
// GPU timestamps the guest asked for (end-of-pipe timestamp writes and reference-clock copies),
// all stamped when the GPU thread parses the packet rather than when the GPU gets there.
inline std::atomic<uint32_t> g_guest_timestamps {0};
// Render passes begun, the largest render area among them, and how many were 3840 wide or more.
inline std::atomic<uint32_t> g_render_passes {0};
inline std::atomic<uint32_t> g_render_max_width {0};
inline std::atomic<uint32_t> g_render_max_height {0};
inline std::atomic<uint32_t> g_render_passes_4k {0};
// Pipeline barriers recorded (both vkCmdPipelineBarrier forms), the image layout transitions
// among them (Image::Transit) and the global barriers guest events ask for (EmitGlobalBarrier).
inline std::atomic<uint32_t> g_barriers {0};
inline std::atomic<uint32_t> g_barriers_image_transit {0};
inline std::atomic<uint32_t> g_barriers_guest_global {0};
// Guest global barriers left out because no work was recorded since the previous one.
inline std::atomic<uint32_t> g_barriers_guest_elided {0};
// Shader resource (SRT) evaluation: full evaluations, cache hits, misses by cause, recorded
// dependency reads re-checked on hits, and verification mismatches (must stay 0).
inline std::atomic<uint32_t> g_srt_evaluations {0};
inline std::atomic<uint32_t> g_srt_hits {0};
inline std::atomic<uint32_t> g_srt_miss_inputs {0};
inline std::atomic<uint32_t> g_srt_miss_memory {0};
inline std::atomic<uint64_t> g_srt_checked_reads {0};
inline std::atomic<uint32_t> g_srt_mismatches {0};
// User data words hashed for SRT cache lookups, and dependency masks widened (tables emptied).
inline std::atomic<uint64_t> g_srt_dependency_words {0};
inline std::atomic<uint32_t> g_srt_mask_resets {0};
// SRT cache hits that substituted changed pass-through descriptor words (inline descriptors).
inline std::atomic<uint32_t> g_srt_substituted {0};
// Guest-CPU reads of GPU-written memory (a game thread faults and the GPU thread reads back):
// count, time waiting for the GPU thread to service the request, time spent reading back, how
// often the buffer's last GPU writer had already completed, and how often the open command buffer
// had to be flushed because the writer was still in it.
inline std::atomic<uint32_t> g_cpu_reads {0};
inline std::atomic<uint64_t> g_cpu_read_service_us {0};
inline std::atomic<uint64_t> g_cpu_read_readback_us {0};
inline std::atomic<uint32_t> g_cpu_read_producer_done {0};
inline std::atomic<uint32_t> g_cpu_read_flushes {0};
// Eager write-back of hot pages: copies scheduled, pages unprotected after write-back, and
// KYTY_DEP_VERIFY mismatches between written-back and GPU data (must stay 0).
inline std::atomic<uint32_t> g_eager_scheduled {0};
inline std::atomic<uint32_t> g_eager_retired {0};
inline std::atomic<uint32_t> g_eager_skipped {0}; // no staging space free without waiting
inline std::atomic<uint32_t> g_dep_mismatches {0};

inline void StoreMax(std::atomic<uint32_t>& target, uint32_t value) {
	auto seen = target.load(std::memory_order_relaxed);
	while (value > seen && !target.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}
inline std::atomic<uint32_t> g_image_creates {0};
inline std::atomic<uint32_t> g_image_deletes {0};
// Diagnostic: wall time spent detiling and uploading new texture data (TextureCache::UploadImage),
// to tell a burst of new-texture stalls apart from shader compiles when both spike together.
inline std::atomic<uint64_t> g_image_upload_us {0};
// Vertex attribute/buffer tables read per draw: copied without faulting because guest memory was
// provably current for their bytes, or loaded directly (which faults on a GPU-modified page).
inline std::atomic<uint32_t> g_vertex_tables_clean {0};
inline std::atomic<uint32_t> g_vertex_tables_direct {0};
// Other shader metadata reads from guest memory (ShaderReadGuest): same split.
inline std::atomic<uint32_t> g_guest_reads_clean {0};
inline std::atomic<uint32_t> g_guest_reads_direct {0};
inline std::atomic<uint32_t> g_clean_read_mismatches {0}; // KYTY_CLEAN_READ_VERIFY=1 only
// Diagnostic: guest-CPU write-protection faults (RenderContext::HandleFault), to tell a burst of
// them apart from shader compiles and texture uploads when several spike on the same frame.
inline std::atomic<uint32_t> g_fault_count {0};
inline std::atomic<uint64_t> g_fault_us {0};

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

// Adds the lifetime of the scope to g_image_upload_us. Separate from CompileScope so a burst of
// new-texture stalls doesn't get attributed to shader compilation.
class UploadScope {
public:
	UploadScope(): m_start(Timer::QueryPerformanceCounter()) {}
	~UploadScope() {
		const auto elapsed = Timer::QueryPerformanceCounter() - m_start;
		g_image_upload_us.fetch_add(elapsed * 1000000 / Timer::QueryPerformanceFrequency(),
		                            std::memory_order_relaxed);
	}

	UploadScope(const UploadScope&)            = delete;
	UploadScope& operator=(const UploadScope&) = delete;

private:
	uint64_t m_start;
};

// Adds the lifetime of the scope to g_fault_us and counts it in g_fault_count.
class FaultScope {
public:
	FaultScope(): m_start(Timer::QueryPerformanceCounter()) {}
	~FaultScope() {
		const auto elapsed = Timer::QueryPerformanceCounter() - m_start;
		g_fault_us.fetch_add(elapsed * 1000000 / Timer::QueryPerformanceFrequency(),
		                     std::memory_order_relaxed);
		g_fault_count.fetch_add(1, std::memory_order_relaxed);
	}

	FaultScope(const FaultScope&)            = delete;
	FaultScope& operator=(const FaultScope&) = delete;

private:
	uint64_t m_start;
};

} // namespace Common::FrameStats

#endif // EMULATOR_SRC_COMMON_FRAMESTATS_H_
