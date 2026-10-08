#include "graphics/host_gpu/renderer/sync.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "kernel/memory.h"
#include "common/timer.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <cstring>
#include <limits>

namespace Libs::Graphics::Sync {

constexpr uint64_t GRAPHICS_REFERENCE_CLOCK_FREQUENCY = 100000000;

bool ScaleReferenceClock(uint64_t host_ticks, uint64_t host_frequency, uint64_t& value) {
	if (host_frequency == 0) {
		return false;
	}

	const auto     whole_seconds = host_ticks / host_frequency;
	const auto     remainder     = host_ticks % host_frequency;
	constexpr auto MAX_VALUE     = std::numeric_limits<uint64_t>::max();
	if (whole_seconds > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY ||
	    remainder > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY) {
		return false;
	}

	const auto whole_value      = whole_seconds * GRAPHICS_REFERENCE_CLOCK_FREQUENCY;
	const auto fractional_value = (remainder * GRAPHICS_REFERENCE_CLOCK_FREQUENCY) / host_frequency;
	if (whole_value > MAX_VALUE - fractional_value) {
		return false;
	}
	value = whole_value + fractional_value;
	return true;
}

uint64_t ReadReferenceClock() {
	const auto host_frequency = LibKernel::KernelGetTscFrequency();
	const auto host_ticks     = LibKernel::KernelReadTsc();
	uint64_t   value          = 0;
	if (!ScaleReferenceClock(host_ticks, host_frequency, value)) {
		EXIT("cannot scale host clock, ticks=0x%016" PRIx64 " frequency=%" PRIu64 "\n", host_ticks,
		     host_frequency);
	}
	return value;
}

namespace {

// GPU timestamps of guest timestamp writes: a ring of queries, each reset on the host after its
// value has been read (hostQueryReset), so recording needs no reset command (which a render pass
// would have to end for).
struct GpuTimestampState {
	vk::Device                     device      = nullptr;
	vk::QueryPool                  pool        = nullptr;
	double                         ns_per_tick = 1.0;
	uint64_t                       valid_mask  = ~uint64_t {0};
	PFN_vkGetCalibratedTimestampsEXT get_calibrated = nullptr;
	uint32_t                       next        = 0;
	std::vector<std::atomic<uint8_t>> busy;
	// Calibration: a GPU timestamp and the host TSC at the same moment.
	std::mutex                     calibration_mutex;
	uint64_t                       calibrated_gpu = 0;
	uint64_t                       calibrated_tsc = 0;
	uint64_t                       calibrated_at  = 0; // TSC of the last calibration
	bool                           disabled       = false;
};

constexpr uint32_t GpuTimestampSlots = 4096;

GpuTimestampState& GpuTimestamps() {
	static GpuTimestampState state;
	return state;
}

bool GpuTimestampsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_TIMESTAMPS");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool InitGpuTimestamps(GraphicContext& graphics) {
	auto& state = GpuTimestamps();
	if (state.pool != nullptr || state.disabled) {
		return state.pool != nullptr;
	}
	state.disabled = true;
	if (!graphics.calibrated_timestamps_enabled || !graphics.host_query_reset_enabled ||
	    graphics.physical_device_properties.limits.timestampComputeAndGraphics == VK_FALSE) {
		return false;
	}
	state.get_calibrated = reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(
	    graphics.device.getProcAddr("vkGetCalibratedTimestampsEXT"));
	if (state.get_calibrated == nullptr) {
		return false;
	}
	uint32_t domain_count = 0;
	const auto get_domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
	    graphics.instance.getProcAddr("vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
	if (get_domains == nullptr ||
	    get_domains(graphics.physical_device, &domain_count, nullptr) != VK_SUCCESS) {
		return false;
	}
	std::vector<VkTimeDomainEXT> domains(domain_count);
	get_domains(graphics.physical_device, &domain_count, domains.data());
	const bool has_device = std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_DEVICE_EXT) !=
	                        domains.end();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto host_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
	const auto host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#endif
	if (!has_device || std::find(domains.begin(), domains.end(), host_domain) == domains.end()) {
		return false;
	}
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = GpuTimestampSlots;
	if (graphics.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
		state.pool = nullptr;
		return false;
	}
	graphics.device.resetQueryPool(state.pool, 0, GpuTimestampSlots);
	state.device      = graphics.device;
	state.ns_per_tick = static_cast<double>(graphics.physical_device_properties.limits.timestampPeriod);
	state.valid_mask  = ~uint64_t {0}; // full 64-bit device timestamps
	state.busy        = std::vector<std::atomic<uint8_t>>(GpuTimestampSlots);
	state.disabled    = false;
	LOGF("GPU timestamps: guest timestamps follow GPU execution (%.3f ns per tick)\n",
	     state.ns_per_tick);
	return true;
}

// Host TSC at the moment of GPU timestamp gpu_ticks, from a calibration at most a second old.
bool GpuTicksToTsc(uint64_t gpu_ticks, uint64_t& tsc) {
	auto&      state     = GpuTimestamps();
	const auto tsc_freq  = LibKernel::KernelGetTscFrequency();
	const auto now_tsc   = LibKernel::KernelReadTsc();
	std::lock_guard lock(state.calibration_mutex);
	if (state.calibrated_at == 0 || now_tsc - state.calibrated_at > tsc_freq) {
		VkCalibratedTimestampInfoEXT infos[2] {};
		infos[0].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
		infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
		infos[1].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		infos[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
		infos[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#endif
		uint64_t stamps[2] {};
		uint64_t deviation = 0;
		if (state.get_calibrated(state.device, 2, infos, stamps, &deviation) != VK_SUCCESS) {
			return false;
		}
		// Map the host sample onto the TSC: read both clocks now and step back.
		const auto host_now = Common::Timer::QueryPerformanceCounter();
		const auto tsc_now  = LibKernel::KernelReadTsc();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const double host_freq = static_cast<double>(Common::Timer::QueryPerformanceFrequency());
		const double host_ago  = static_cast<double>(host_now) - static_cast<double>(stamps[1]);
#else
		// CLOCK_MONOTONIC_RAW nanoseconds; Timer::QueryPerformanceCounter uses the same clock here.
		const double host_freq = static_cast<double>(Common::Timer::QueryPerformanceFrequency());
		const double host_ago  = (static_cast<double>(host_now) / host_freq -
		                          static_cast<double>(stamps[1]) / 1e9) * host_freq;
#endif
		state.calibrated_gpu = stamps[0] & state.valid_mask;
		state.calibrated_tsc = tsc_now - static_cast<uint64_t>(std::max(0.0, host_ago) *
		                                                        static_cast<double>(tsc_freq) /
		                                                        host_freq);
		state.calibrated_at  = tsc_now;
	}
	const auto   delta_ticks = static_cast<int64_t>(gpu_ticks - state.calibrated_gpu);
	const double delta_ns    = static_cast<double>(delta_ticks) * state.ns_per_tick;
	const double value       = static_cast<double>(state.calibrated_tsc) +
	                     delta_ns * static_cast<double>(tsc_freq) / 1e9;
	if (value <= 0.0) {
		return false;
	}
	tsc = static_cast<uint64_t>(value);
	return true;
}

} // namespace

void RecordGpuTimestamp(CommandBuffer& buffer, void* dst, bool bottom_of_pipe) {
	if (!GpuTimestampsEnabled() || dst == nullptr || !InitGpuTimestamps(buffer.GetGraphics())) {
		return;
	}
	auto&      state = GpuTimestamps();
	const auto slot  = state.next;
	if (state.busy[slot].exchange(1u, std::memory_order_acq_rel) != 0u) {
		return; // the ring is full of unread timestamps: keep the processing-time value
	}
	state.next = (state.next + 1) % GpuTimestampSlots;
	buffer.Handle().writeTimestamp(bottom_of_pipe ? vk::PipelineStageFlagBits::eBottomOfPipe
	                                              : vk::PipelineStageFlagBits::eTopOfPipe,
	                               state.pool, slot);
	auto& scheduler = buffer.GetContext().GetCommandScheduler();
	scheduler.DeferPriorityOperation([slot, dst] {
		auto&    state = GpuTimestamps();
		uint64_t ticks = 0;
		const auto result = state.device.getQueryPoolResults(
		    state.pool, slot, 1, sizeof(ticks), &ticks, sizeof(ticks), vk::QueryResultFlagBits::e64);
		uint64_t tsc   = 0;
		uint64_t value = 0;
		if (result == vk::Result::eSuccess && GpuTicksToTsc(ticks & state.valid_mask, tsc) &&
		    ScaleReferenceClock(tsc, LibKernel::KernelGetTscFrequency(), value)) {
			(void)LibKernel::Memory::TryWriteBacking(reinterpret_cast<uint64_t>(dst), &value, sizeof(value));
		}
		state.device.resetQueryPool(state.pool, slot, 1);
		state.busy[slot].store(0u, std::memory_order_release);
	});
}

enum class EndOfPipeWriteSize : uint32_t { Dword = 4, Qword = 8 };
enum class EndOfPipeWriteAction { Write, WriteBack, Interrupt, InterruptWriteBack };

static CommandBufferDebugOp DebugOperation(EndOfPipeWriteAction action) {
	switch (action) {
		case EndOfPipeWriteAction::Write: return CommandBufferDebugOp::EopWrite;
		case EndOfPipeWriteAction::WriteBack:
		case EndOfPipeWriteAction::InterruptWriteBack: return CommandBufferDebugOp::EopWriteBack;
		case EndOfPipeWriteAction::Interrupt: return CommandBufferDebugOp::EopInterrupt;
	}
	EXIT("unsupported end-of-pipe write action\n");
	return CommandBufferDebugOp::Unknown;
}

static bool TriggersInterrupt(EndOfPipeWriteAction action) {
	return action == EndOfPipeWriteAction::Interrupt ||
	       action == EndOfPipeWriteAction::InterruptWriteBack;
}

static void RecordEndOfPipeWrite(uint64_t submit_id, CommandBuffer& buffer, uint64_t destination,
                                 uint64_t value, EndOfPipeWriteSize size,
                                 EndOfPipeWriteAction action, int interrupt_event_id = 0,
                                 uint32_t context_id = 0) {
	EXIT_IF(destination == 0);
	(void)buffer.Handle();

	const auto width      = static_cast<uint32_t>(size);
	const auto value_low  = static_cast<uint32_t>(value);
	const auto value_high = static_cast<uint32_t>(value >> 32u);
	const auto operation  = static_cast<uint32_t>(DebugOperation(action));
	if (TriggersInterrupt(action)) {
		buffer.SetDebugInfo(operation, submit_id, width, context_id, value_low, value_high,
		                    destination);
		TriggerEopEventAtEndOfPipe(buffer, interrupt_event_id, context_id);
	} else {
		buffer.SetDebugInfo(operation, submit_id, width, value_low, value_high, 0, destination);
	}
}

void WriteAtEndOfPipe32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                        uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeGds32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                           uint32_t dw_offset, uint32_t dw_num) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWrite), submit_id,
	                    dw_offset, dw_num, 0, 0, reinterpret_cast<uint64_t>(dst_gpu_addr));
}

void WriteAtEndOfPipe64(uint64_t submit_id, CommandBuffer& buffer, uint64_t* dst_gpu_addr,
                        uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeWithWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithInterruptWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                              uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterruptWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                              uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterrupt64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

void WriteAtEndOfPipeWithInterrupt32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

uint64_t PrepareVideoOutFlip(CommandBuffer& buffer, int handle, int index, int flip_mode,
                             int64_t flip_arg) {
	for (;;) {
		uint64_t   request_id = 0;
		auto&      video_out  = buffer.GetContext().GetVideoOut();
		const auto result =
		    video_out.SubmitFlipFromGpu(buffer, handle, index, flip_mode, flip_arg, request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			return request_id;
		}
		if (result != VideoOut::VIDEO_OUT_ERROR_FLIP_QUEUE_FULL) {
			EXIT("GPU flip submission failed, result=%d handle=%d index=%d mode=%d arg=%" PRId64
			     "\n",
			     result, handle, index, flip_mode, flip_arg);
		}
		video_out.WaitForSubmitSlot(handle);
	}
}

void WriteAtEndOfPipeWithInterruptWriteBackFlip32(uint64_t submit_id, CommandBuffer& buffer,
                                                  uint32_t* dst_gpu_addr, uint32_t value,
                                                  int handle, int index, int flip_mode,
                                                  int64_t flip_arg, uint64_t request_id,
                                                  int event_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWriteBackFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation([&renderer, event_id, request_id] {
		renderer.GetVideoOut().CompleteFlip(request_id);
		renderer.TriggerInterrupt(event_id, 0);
	});
}

void WriteAtEndOfPipeWithFlip32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                                uint32_t value, int handle, int index, int flip_mode,
                                int64_t flip_arg, uint64_t request_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void WriteAtEndOfPipeOnlyFlip(uint64_t submit_id, CommandBuffer& buffer, int handle, int index,
                              int flip_mode, int64_t flip_arg, uint64_t request_id) {
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopOnlyFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), 0, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void TriggerEopEventAtEndOfPipe(CommandBuffer& buffer, int event_id, uint32_t context_id) {
	(void)buffer.Handle();
	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, event_id, context_id] { renderer.TriggerInterrupt(event_id, context_id); });
}

static void InterruptEventResetFunc(LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void InterruptEventTriggerFunc(LibKernel::EventQueue::KernelEqueueEvent* event,
                                      void*                                     trigger_data) {
	EXIT_IF(event == nullptr);

	// kqueue delivers a registration at most once per retrieval: interrupts that arrive while it
	// is still pending coalesce into it, fflags counting them (reset on retrieval) and data
	// carrying the latest one's value. Queueing a copy per interrupt instead grew without bound
	// (Astro Bot drains its EOP queues at ~350 events/s while ~13,000/s are raised) and handed
	// the waiting threads a backlog of stale interrupts instead of letting them sleep.
	event->event.fflags++;
	event->event.data = reinterpret_cast<intptr_t>(trigger_data);
	event->triggered  = true;
}

int AddEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id,
               void* udata) {
	LibKernel::EventQueue::KernelEqueueEvent event;
	event.triggered           = false;
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.event.udata         = udata;
	event.event.fflags        = 0;
	event.event.data          = id;
	event.filter.reset_func   = InterruptEventResetFunc;
	event.filter.trigger_func = InterruptEventTriggerFunc;

	int result = LibKernel::EventQueue::KernelAddEvent(eq, event);

	if (result == 0) {
		renderer.AddInterruptEq(eq, id);
	}

	return result;
}

int DeleteEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id) {
	int result = LibKernel::EventQueue::KernelDeleteEvent(
	    eq, static_cast<uintptr_t>(id), LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);
	if (result == OK || result == LibKernel::KERNEL_ERROR_ENOENT) {
		renderer.DeleteInterruptEq(eq, id);
	}

	return result;
}

void ReadGds(const Buffer& gds, uint32_t* dst, uint32_t dw_offset, uint32_t dw_size) {
	const auto offset = uint64_t {dw_offset} * sizeof(uint32_t);
	const auto size   = uint64_t {dw_size} * sizeof(uint32_t);
	EXIT_IF(dst == nullptr || offset > gds.Size() || size > gds.Size() - offset ||
	        gds.Mapped().empty());
	std::memcpy(dst, gds.Mapped().data() + offset, static_cast<size_t>(size));
}

} // namespace Libs::Graphics::Sync
