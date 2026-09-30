#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/gpuCheckpoints.h"

#include "common/assert.h"
#include "common/gpuWaitDiagnostics.h"
#include "graphics/host_gpu/graphicContext.h"

#include <cstdio>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	WaitTimeline(m_graphics, m_semaphore, tick, "GPU work", CurrentTick());
	Refresh();
}

void WaitTimeline(GraphicContext& graphics, vk::Semaphore semaphore, uint64_t value,
                  const char* what, uint64_t next_unsubmitted) {
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &semaphore;
	wait_info.pValues        = &value;

	// An unbounded vkWaitSemaphores does not return on this driver once the device is lost (a GPU
	// hang the OS reset), which froze the emulator for good. Wait in slices instead: while the GPU
	// is alive this is the same wait, and a lost device is noticed and reported.
	const bool     diagnostics = Common::GpuWaitDiagnostics::Enabled();
	const uint64_t slice_ns    = diagnostics ? 2'000'000'000ull : 1'000'000'000ull;
	vk::Result     result      = vk::Result::eSuccess;
	for (uint32_t slice = 1;; slice++) {
		result = graphics.device.waitSemaphores(&wait_info, slice_ns);
		if (result != vk::Result::eTimeout) {
			break;
		}
		uint64_t   counter = 0;
		const auto state   = graphics.device.getSemaphoreCounterValue(semaphore, &counter);
		if (state == vk::Result::eErrorDeviceLost) {
			result = state;
			break;
		}
		if (diagnostics) {
			char header[256];
			std::snprintf(header, sizeof(header),
			              "timeline %p (%s): waiting %u s for value %llu, GPU reached %llu, next "
			              "unsubmitted %llu",
			              static_cast<void*>(static_cast<VkSemaphore>(semaphore)), what,
			              static_cast<unsigned>(slice * slice_ns / 1'000'000'000ull),
			              static_cast<unsigned long long>(value),
			              static_cast<unsigned long long>(counter),
			              static_cast<unsigned long long>(next_unsubmitted));
			if (slice == 1 || slice % 5 == 0) {
				Common::GpuWaitDiagnostics::Dump(header);
			} else {
				std::printf("[gpu-wait] %s\n", header);
				std::fflush(stdout);
			}
		}
	}
	if (result == vk::Result::eErrorDeviceLost) {
		GpuCheckpoints::ReportDeviceLost(graphics.queue);
		if (diagnostics) {
			Common::GpuWaitDiagnostics::Dump("device lost while waiting for the GPU");
		}
		EXIT("GPU device lost while waiting for %s (value %llu): the GPU stopped responding and "
		     "the driver reset it. Run with KYTY_GPU_WAIT_DIAGNOSTICS=1 to identify the work that "
		     "hung it.\n",
		     what, static_cast<unsigned long long>(value));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

} // namespace Libs::Graphics
