#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/cache/bufferDownloadBatch.h"

#include "common/gpuWaitDiagnostics.h"
#include "common/alignment.h"
#include "common/frameStats.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto&                buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

static bool GpuHazardVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_HAZARD_VERIFY");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void BufferCache::VerifyBindingAlive(vk::Buffer handle, const char* use, uint64_t submit_id) {
	if (!GpuHazardVerifyEnabled() || !handle) {
		return;
	}
	const auto it = m_retired_buffers.find(static_cast<VkBuffer>(handle));
	if (it == m_retired_buffers.end() || it->second.tick >= m_scheduler.CurrentTick()) {
		return;
	}
	static uint64_t hazards = 0;
	if (++hazards <= 200 || hazards % 1000 == 0) {
		m_scheduler.GetMasterSemaphore().Refresh();
		std::printf("[gpu-hazard] %s binds buffer guest=0x%" PRIx64 "+0x%" PRIx64
		            " deleted in tick %" PRIu64 ", bound in tick %" PRIu64
		            " (deleted tick complete=%d) submit=%" PRIu64 " count=%" PRIu64 "\n",
		            use, it->second.guest, it->second.size, it->second.tick,
		            m_scheduler.CurrentTick(),
		            m_scheduler.GetMasterSemaphore().IsFree(it->second.tick) ? 1 : 0, submit_id,
		            hazards);
		std::fflush(stdout);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		if (GpuHazardVerifyEnabled()) {
			const auto& buffer = m_slot_buffers[id];
			m_retired_buffers[static_cast<VkBuffer>(buffer.Handle())] = {
			    m_scheduler.CurrentTick(), buffer.CpuAddress(), buffer.Size()};
		}
		m_scheduler.DeferOperation([this, id] {
			if (GpuHazardVerifyEnabled()) {
				m_retired_buffers.erase(static_cast<VkBuffer>(m_slot_buffers[id].Handle()));
			}
			if (m_graphics.device_address_destruction_waits_for_queue) {
				// Every command buffer submitted so far may reference this device-address buffer.
				// ponytail: drains the queue per deletion batch; a per-buffer retire tick would
				// need the driver to stop referencing buffers the work does not use.
				m_scheduler.GetMasterSemaphore().Wait(m_scheduler.CurrentTick() - 1);
			}
			m_slot_buffers.erase(id);
		});
	} else {
		m_slot_buffers.erase(id);
	}
}

namespace {

std::vector<BufferDownloadRange> ToDownloadRanges(const std::vector<vk::BufferCopy>& copies) {
	std::vector<BufferDownloadRange> ranges;
	ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		ranges.push_back({copy.srcOffset, copy.size});
	}
	return ranges;
}

} // namespace

bool BufferCache::CollectDownloadCopies(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                        std::vector<vk::BufferCopy>& copies,
                                        uint64_t&                    total_size) {
	const auto buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	return !copies.empty();
}

bool BufferCache::ReadbackSubmitted(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> collected;
	uint64_t                    total_size = 0;
	if (!CollectDownloadCopies(buffer, vaddr, size, collected, total_size)) {
		return false;
	}

	const auto device = m_graphics.device;
	if (m_readback_pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.queue_family;
		EXIT_IF(device.createCommandPool(&pool_info, nullptr, &m_readback_pool) !=
		        vk::Result::eSuccess);
		vk::CommandBufferAllocateInfo alloc_info {};
		alloc_info.commandPool        = m_readback_pool;
		alloc_info.level              = vk::CommandBufferLevel::ePrimary;
		alloc_info.commandBufferCount = 1;
		EXIT_IF(device.allocateCommandBuffers(&alloc_info, &m_readback_command) !=
		        vk::Result::eSuccess);
		vk::SemaphoreTypeCreateInfo type_info {};
		type_info.semaphoreType = vk::SemaphoreType::eTimeline;
		vk::SemaphoreCreateInfo semaphore_info {};
		semaphore_info.pNext = &type_info;
		EXIT_IF(device.createSemaphore(&semaphore_info, nullptr, &m_readback_semaphore) !=
		        vk::Result::eSuccess);
	}

	// Stage in batches that each fit one download-buffer reservation, as DownloadBufferMemory
	// does. Every batch is submitted and waited on before the next is mapped.
	const auto buffer_address = buffer.CpuAddress();
	for (const auto& batch: SplitBufferDownload(ToDownloadRanges(collected),
	                                            m_download_buffer.MaxReservation(), 64)) {
		const auto [mapped, offset] = m_download_buffer.Map(batch.total_size, 64);
		Common::FrameStats::g_readback_bytes.fetch_add(batch.total_size, std::memory_order_relaxed);
		if (mapped == nullptr) {
			EXIT("BufferCache: readback batch could not be staged\n");
		}
		m_download_buffer.Commit();
		std::vector<vk::BufferCopy> copies;
		copies.reserve(batch.copies.size());
		for (const auto& copy: batch.copies) {
			copies.push_back({copy.src_offset, copy.dst_offset + offset, copy.size});
		}

		const auto command = m_readback_command;
		command.reset({});
		vk::CommandBufferBeginInfo begin_info {};
		begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		EXIT_IF(command.begin(&begin_info) != vk::Result::eSuccess);
		// A barrier's first scope covers everything submitted earlier to this queue, including
		// the already submitted command buffers that wrote this range.
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                        nullptr);
		command.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
		                   static_cast<uint32_t>(copies.size()), copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eHostRead;
		after.buffer        = m_download_buffer.Handle();
		after.offset        = offset;
		after.size          = batch.total_size;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
		                        nullptr);
		EXIT_IF(command.end() != vk::Result::eSuccess);

		const uint64_t                  signal_value = ++m_readback_tick;
		Common::GpuWaitDiagnostics::Note("readback-submit", signal_value, batch.total_size);
		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.signalSemaphoreValueCount = 1;
		timeline_info.pSignalSemaphoreValues    = &signal_value;
		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &command;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &m_readback_semaphore;
		{
			Common::LockGuard lock(m_graphics.queue_mutex);
			const auto        result = m_graphics.queue.submit(1, &submit_info, nullptr);
			if (result != vk::Result::eSuccess) {
				if (result == vk::Result::eErrorDeviceLost) {
					GpuCheckpoints::ReportDeviceLost(m_graphics.queue);
				}
				EXIT("BufferCache: readback submit failed: %s\n", vk::to_string(result).c_str());
			}
		}
		const auto wait_start = Common::Timer::QueryPerformanceCounter();
		// Reports a lost device instead of waiting forever (see WaitTimeline).
		WaitTimeline(m_graphics, m_readback_semaphore, signal_value, "a buffer readback");
		const auto wait_result = vk::Result::eSuccess;
		Common::FrameStats::g_readback_count.fetch_add(1, std::memory_order_relaxed);
		Common::FrameStats::g_readback_wait_us.fetch_add(
		    (Common::Timer::QueryPerformanceCounter() - wait_start) * 1000000 /
		        Common::Timer::QueryPerformanceFrequency(),
		    std::memory_order_relaxed);
		if (wait_result != vk::Result::eSuccess) {
			if (wait_result == vk::Result::eErrorDeviceLost) {
				GpuCheckpoints::ReportDeviceLost(m_graphics.queue);
			}
			EXIT("BufferCache: readback wait failed: %s\n", vk::to_string(wait_result).c_str());
		}

		m_download_buffer.Invalidate(offset, batch.total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	}
	return true;
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> collected;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	if (!CollectDownloadCopies(buffer, vaddr, size, collected, total_size)) {
		return false;
	}

	// StreamBuffer::Map refuses a single reservation larger than the staging buffer instead of
	// waiting for space, so a download whose dirty ranges total more than one staging pass
	// could never be mapped and used to abort the process here. Stage it in batches instead:
	// each reserves its own region, and the stream keeps that region reserved until its
	// deferred write-back has consumed it, so consecutive batches cannot overlap.
	for (const auto& batch: SplitBufferDownload(ToDownloadRanges(collected),
	                                            m_download_buffer.MaxReservation(), 64)) {
		const auto [mapped, offset] = m_download_buffer.Map(batch.total_size, 64);
		if (mapped == nullptr) {
			EXIT("BufferCache: download batch could not be staged\n");
		}
		m_download_buffer.Commit();
		// Map can wait for a pending use of the staging buffer to retire. Waiting submits the
		// current command buffer and begins a new one, so re-read the handle after every Map
		// instead of recording later batches into a handle captured before the loop.
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		std::vector<vk::BufferCopy> copies;
		copies.reserve(batch.copies.size());
		for (const auto& copy: batch.copies) {
			copies.push_back({copy.src_offset, copy.dst_offset + offset, copy.size});
		}

		vk::BufferMemoryBarrier before {};
		before.srcAccessMask =
		    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                       nullptr);
		native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
		                  static_cast<uint32_t>(copies.size()), copies.data());

		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eHostRead;
		after.buffer        = m_download_buffer.Handle();
		after.offset        = offset;
		after.size          = batch.total_size;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands |
		                           vk::PipelineStageFlagBits::eHost,
		                       {}, 0, nullptr, 1, &after, 0, nullptr);
		const uint64_t total = batch.total_size;
		m_scheduler.DeferPriorityOperation([this, mapped, offset, total, buffer_address,
		                                    copies = std::move(copies)] {
			m_download_buffer.Invalidate(offset, total);
			for (const auto& copy: copies) {
				const auto* const staged = mapped + (copy.dstOffset - offset);
				Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset, staged,
				                                      copy.size);
			}
		});
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
	if (m_readback_pool != nullptr) {
		m_graphics.device.destroySemaphore(m_readback_semaphore, nullptr);
		m_graphics.device.destroyCommandPool(m_readback_pool, nullptr);
	}
}

void BufferCache::ForgetKnownFills(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_known_fills_mutex);
	const uint64_t  end = vaddr + size;
	auto            it  = m_known_fills.lower_bound(vaddr);
	if (it != m_known_fills.begin() && std::prev(it)->second.end > vaddr) {
		--it;
	}
	while (it != m_known_fills.end() && it->first < end) {
		const auto start = it->first;
		const auto fill  = it->second;
		it               = m_known_fills.erase(it);
		// Keep the parts of a fill outside the forgotten range.
		if (start < vaddr) {
			m_known_fills.emplace(start, KnownFill {vaddr, fill.value});
		}
		if (fill.end > end) {
			it = m_known_fills.emplace(end, KnownFill {fill.end, fill.value}).first;
			++it;
		}
	}
}

void BufferCache::RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	ForgetKnownFills(vaddr, size);
	std::lock_guard lock(m_known_fills_mutex);
	m_known_fills[vaddr] = KnownFill {vaddr + size, value};
}

bool BufferCache::TryGetKnownFill(uint64_t vaddr, uint64_t size, uint32_t* value) {
	std::lock_guard lock(m_known_fills_mutex);
	auto            it = m_known_fills.upper_bound(vaddr);
	if (it == m_known_fills.begin()) {
		return false;
	}
	--it;
	if (it->first > vaddr || it->second.end < vaddr + size) {
		return false;
	}
	*value = it->second.value;
	return true;
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	ForgetKnownFills(vaddr, size);
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const bool guest_cpu     = !GuestGpu::IsGpuThread();
	const auto request_start = Common::Timer::QueryPerformanceCounter();
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, guest_cpu,
	                                                request_start] {
		const auto service_start = Common::Timer::QueryPerformanceCounter();
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (guest_cpu) {
			m_scheduler.GetMasterSemaphore().Refresh();
			const auto freq = Common::Timer::QueryPerformanceFrequency();
			Common::FrameStats::g_cpu_reads.fetch_add(1, std::memory_order_relaxed);
			Common::FrameStats::g_cpu_read_service_us.fetch_add(
			    (service_start - request_start) * 1000000 / freq, std::memory_order_relaxed);
			if (m_scheduler.GetMasterSemaphore().IsFree(buffer.last_gpu_write_tick)) {
				Common::FrameStats::g_cpu_read_producer_done.fetch_add(1, std::memory_order_relaxed);
			}
			if (buffer.last_gpu_write_tick >= m_scheduler.CurrentTick()) {
				Common::FrameStats::g_cpu_read_flushes.fetch_add(1, std::memory_order_relaxed);
			}
		}

		// Widen nearby CPU reads so they share one GPU drain.
		constexpr uint64_t WindowSize   = 512 * 1024;
		const auto         buffer_begin = buffer.CpuAddress();
		const auto         buffer_end   = buffer_begin + buffer.Size();
		const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
		const auto window_end =
		    std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

		if (guest_cpu && m_hot_pages.size() < 256) {
			m_hot_pages.try_emplace(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), 0);
		}
		// An eager write-back in flight has already taken its bytes out of the dirty ranges, so
		// finish it (and unprotect what it covered) before looking for anything left to read.
		RetireHotPages(window_begin, window_end - window_begin, true);

		// Large reads (PPSA10595's stage select) exceed the 64 MiB staging buffer, so read back in
		// chunks that each complete before the next reuses the staging space. Once one chunk has
		// submitted the recording command buffer, the rest take the submitted-work path.
		// Experiment switch: KYTY_SLOW_READBACK=1 always drains the recording command buffer.
		static const bool  slow_readback = std::getenv("KYTY_SLOW_READBACK") != nullptr;
		constexpr uint64_t ChunkSize     = 8 * 1024 * 1024;
		for (uint64_t chunk = window_begin; chunk < window_end; chunk += ChunkSize) {
			const auto chunk_size = std::min(ChunkSize, window_end - chunk);
			if (!slow_readback && buffer.last_gpu_write_tick < m_scheduler.CurrentTick()) {
				if (ReadbackSubmitted(buffer, chunk, chunk_size)) {
					m_memory_tracker.UnmarkRegionAsGpuModified(chunk, chunk_size);
				}
			} else if (DownloadBufferMemory(buffer, chunk, chunk_size)) {
				const auto tick = m_scheduler.CurrentTick();
				m_scheduler.Wait(tick);
				m_scheduler.WaitPriorityOperations(tick);
				m_memory_tracker.UnmarkRegionAsGpuModified(chunk, chunk_size);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		if (guest_cpu) {
			Common::FrameStats::g_cpu_read_readback_us.fetch_add(
			    (Common::Timer::QueryPerformanceCounter() - service_start) * 1000000 /
			        Common::Timer::QueryPerformanceFrequency(),
			    std::memory_order_relaxed);
		}
	});
}

static bool DepVerifyEnabled() {
	static const bool enabled = std::getenv("KYTY_DEP_VERIFY") != nullptr;
	return enabled;
}

static bool EagerWriteBackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EAGER_WRITEBACK");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void BufferCache::WriteBackHotPages() {
	if (m_hot_pages.empty() || !EagerWriteBackEnabled()) {
		return;
	}
	RetireHotPages(0, UINT64_MAX, false);
	for (auto& [page, inflight]: m_hot_pages) {
		if (inflight != 0 || !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
			continue;
		}
		const auto id = FindBuffer(page, 1);
		if (IsBufferInvalid(id)) {
			continue;
		}
		auto&      buffer = m_slot_buffers[id];
		const auto begin  = std::max(page, buffer.CpuAddress());
		const auto end    = std::min(page + TRACKER_PAGE_SIZE, buffer.CpuAddress() + buffer.Size());
		if (begin < end && EagerDownload(buffer, begin, end - begin)) {
			// The copy and its deferred write-back belong to the command buffer about to be
			// submitted as this tick.
			inflight = m_scheduler.CurrentTick();
			Common::FrameStats::g_eager_scheduled.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

// DownloadBufferMemory for an eager write-back, which is only an optimization and so must never
// block: staging space is reserved without waiting before any dirty range is taken, and when none
// is free the page is simply left for the next submit (or an on-demand read) to handle.
bool BufferCache::EagerDownload(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	uint64_t needed = 0;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		needed += Common::AlignUp(end - start, 64);
	});
	if (needed == 0) {
		return false;
	}
	const auto [mapped, offset] = m_download_buffer.Map(needed, 64, false);
	if (mapped == nullptr) {
		Common::FrameStats::g_eager_skipped.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total = 0;
	if (!CollectDownloadCopies(buffer, vaddr, size, copies, total)) {
		return false;
	}
	EXIT_IF(total > needed);
	for (const auto& copy: copies) {
		m_eager_pending_ranges.Add(buffer.CpuAddress() + copy.srcOffset, copy.size);
	}
	m_download_buffer.Commit();
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = needed;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	const auto buffer_address = buffer.CpuAddress();
	m_scheduler.DeferPriorityOperation([this, mapped, offset, needed, buffer_address,
	                                    copies = std::move(copies)] {
		m_download_buffer.Invalidate(offset, needed);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	});
	return true;
}

void BufferCache::RetireHotPages(uint64_t vaddr, uint64_t size, bool wait) {
	auto&      master = m_scheduler.GetMasterSemaphore();
	const auto end    = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	master.Refresh();
	for (auto it = m_hot_pages.lower_bound(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
	     it != m_hot_pages.end() && it->first < end; ++it) {
		auto& [page, inflight] = *it;
		if (inflight == 0) {
			continue;
		}
		if (!master.IsFree(inflight) || !m_scheduler.PriorityOperationsDone(inflight)) {
			if (!wait) {
				continue;
			}
			m_scheduler.Wait(inflight);
			m_scheduler.WaitPriorityOperations(inflight);
		}
		inflight = 0;
		m_eager_pending_ranges.Subtract(page, TRACKER_PAGE_SIZE); // written to guest memory
		// A newer GPU write landed after the copy was recorded: stay protected; its bytes are
		// written back by a later pass or read back on demand.
		if (m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
			continue;
		}
		if (DepVerifyEnabled() && !VerifyWrittenBack(page)) {
			continue;
		}
		m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
		Common::FrameStats::g_eager_retired.fetch_add(1, std::memory_order_relaxed);
	}
}

// KYTY_DEP_VERIFY: before unprotecting a written-back page, read the same bytes straight from the
// GPU buffer (a full synchronous readback, the old conservative path) and compare.
bool BufferCache::VerifyWrittenBack(uint64_t page) {
	const auto id = FindBuffer(page, 1);
	if (IsBufferInvalid(id)) {
		return true;
	}
	auto&      buffer = m_slot_buffers[id];
	const auto begin  = std::max(page, buffer.CpuAddress());
	const auto end    = std::min(page + TRACKER_PAGE_SIZE, buffer.CpuAddress() + buffer.Size());
	const auto size   = end - begin;
	m_scheduler.Finish();
	const auto [mapped, offset] = m_download_buffer.Map(size, 64);
	EXIT_IF(mapped == nullptr);
	m_download_buffer.Commit();
	auto&                   command = m_scheduler.Current();
	vk::BufferCopy          copy {begin - buffer.CpuAddress(), offset, size};
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	barrier.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &barrier, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(), 1, &copy);
	m_scheduler.Finish();
	m_download_buffer.Invalidate(offset, size);
	std::vector<uint8_t> backing(size);
	if (!Libs::LibKernel::Memory::TryReadBacking(begin, backing.data(), size)) {
		return true;
	}
	if (std::memcmp(backing.data(), mapped, size) == 0) {
		return true;
	}
	size_t first = 0;
	while (first < size && backing[first] == mapped[first]) {
		++first;
	}
	Common::FrameStats::g_dep_mismatches.fetch_add(1, std::memory_order_relaxed);
	std::printf("GPU DEPENDENCY MISMATCH: page=0x%016" PRIx64 " buffer=[0x%016" PRIx64
	            ",+0x%" PRIx64 ") first_diff=+0x%zx written_back=0x%02x gpu=0x%02x "
	            "last_gpu_write_tick=%" PRIu64 " gpu_done_tick=%" PRIu64 "\n",
	            page, buffer.CpuAddress(), buffer.Size(), first, backing[first], mapped[first],
	            buffer.last_gpu_write_tick, m_scheduler.GetMasterSemaphore().KnownGpuTick());
	return false;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice
			// versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end     = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr              = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	// Merged contents are copied by the command buffer being recorded.
	m_slot_buffers[id].last_gpu_write_tick = m_scheduler.CurrentTick();
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			std::memcpy(mapped + copy.srcOffset, reinterpret_cast<const void*>(address), copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                          vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		std::memcpy(temporary->Mapped().data() + copy.srcOffset,
		            reinterpret_cast<const void*>(address), copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

static bool BufferVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_VERIFY");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void BufferCache::VerifyCoherence(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// Diagnostics only. Bounded per tick so a heavy frame stays playable while verifying.
	constexpr uint64_t MaxRangeBytes = 4 * 1024 * 1024;
	constexpr uint64_t MaxTickBytes  = 32 * 1024 * 1024;
	struct Stats {
		uint64_t tick = 0, tick_bytes = 0, ranges = 0, bytes = 0, mismatches = 0, skipped = 0;
		uint64_t last_report = 0;
	};
	static Stats stats;
	const auto   tick = m_scheduler.CurrentTick();
	if (stats.tick != tick) {
		stats.tick       = tick;
		stats.tick_bytes = 0;
	}
	if (size == 0 || size > MaxRangeBytes || stats.tick_bytes + size > MaxTickBytes) {
		stats.skipped++;
		return;
	}

	// Snapshot first, then classify: a page still clean after the snapshot was not written since
	// its last upload, because the first write to a clean page faults and marks it dirty before
	// the write lands. The GPU copy must equal the snapshot on such pages.
	std::vector<uint8_t> guest(size);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, guest.data(), size)) {
		stats.skipped++;
		return;
	}
	struct Piece {
		uint64_t offset, bytes;
	};
	std::vector<Piece> clean;
	for (uint64_t address = vaddr; address < vaddr + size;) {
		const auto next  = std::min(Common::AlignDown(address, TRACKER_PAGE_SIZE) + TRACKER_PAGE_SIZE,
		                            vaddr + size);
		const auto bytes = next - address;
		const bool dirty = m_memory_tracker.IsRegionCpuModified(address, bytes) ||
		                   m_memory_tracker.IsRegionGpuModified(address, bytes) ||
		                   m_gpu_modified_ranges.Intersects(address, bytes) ||
		                   m_eager_pending_ranges.Intersects(address, bytes) ||
		                   m_texture_cache.IsRegionGpuModified(address, bytes);
		if (!dirty) {
			if (!clean.empty() && clean.back().offset + clean.back().bytes == address - vaddr) {
				clean.back().bytes += bytes;
			} else {
				clean.push_back({address - vaddr, bytes});
			}
		}
		address = next;
	}
	if (clean.empty()) {
		stats.skipped++;
		return;
	}
	stats.tick_bytes += size;

	auto readback = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         vk::BufferUsageFlagBits::eTransferDst, size);
	readback->CopyFrom(m_scheduler.Current(), buffer, buffer.Offset(vaddr), 0, size,
	                   vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eHostRead);
	m_scheduler.DeferOperation([readback = std::move(readback), guest = std::move(guest),
	                            clean = std::move(clean), vaddr, size, tick,
	                            buffer_base = buffer.CpuAddress()]() mutable {
		readback->Invalidate(0, size);
		const auto* gpu = readback->Mapped().data();
		stats.ranges++;
		stats.bytes += size;
		for (const auto& piece: clean) {
			if (std::memcmp(gpu + piece.offset, guest.data() + piece.offset, piece.bytes) == 0) {
				continue;
			}
			uint64_t first = piece.offset, differing = 0;
			while (gpu[first] == guest[first]) {
				first++;
			}
			for (uint64_t i = piece.offset; i < piece.offset + piece.bytes; i++) {
				differing += gpu[i] != guest[i] ? 1 : 0;
			}
			if (++stats.mismatches <= 64) {
				uint32_t gpu_words[4] {}, guest_words[4] {};
				const auto aligned = first & ~uint64_t {3};
				const auto avail   = std::min<uint64_t>(sizeof(gpu_words), size - aligned);
				std::memcpy(gpu_words, gpu + aligned, avail);
				std::memcpy(guest_words, guest.data() + aligned, avail);
				std::printf("[buf-verify] STALE range=0x%" PRIx64 "+0x%" PRIx64 " buffer=0x%" PRIx64
				            " tick=%" PRIu64 " at=0x%" PRIx64 " clean_piece=+0x%" PRIx64
				            "+0x%" PRIx64 " differing=%" PRIu64
				            " gpu=%08x %08x %08x %08x guest=%08x %08x %08x %08x\n",
				            vaddr, size, buffer_base, tick, vaddr + first, piece.offset, piece.bytes,
				            differing, gpu_words[0], gpu_words[1], gpu_words[2], gpu_words[3],
				            guest_words[0], guest_words[1], guest_words[2], guest_words[3]);
				std::fflush(stdout);
			}
			break;
		}
		if (stats.ranges - stats.last_report >= 20000) {
			stats.last_report = stats.ranges;
			std::printf("[buf-verify] checked=%" PRIu64 " bytes=%" PRIu64 " stale=%" PRIu64
			            " skipped=%" PRIu64 "\n",
			            stats.ranges, stats.bytes, stats.mismatches, stats.skipped);
			std::fflush(stdout);
		}
	});
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (!is_written && BufferVerifyEnabled()) {
		VerifyCoherence(buffer, vaddr, size);
	}
	if (is_written) {
		if (Common::GpuWaitDiagnostics::Enabled()) {
			if (m_writers.size() > 200000) {
				m_writers.clear();
			}
			m_writers[vaddr] = {size, Common::GpuWaitDiagnostics::CurrentOp(),
			                    m_scheduler.CurrentTick()};
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
		ForgetKnownFills(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		// Map refuses a reservation larger than the staging buffer instead of waiting for
		// space, so an oversized image upload cannot use the shared one. Copy the guest data
		// into a private temporary of exactly this size instead, and let its deferred release
		// retire the buffer after the upload, as UploadCopies already does for buffer uploads.
		auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
		                                         vk::BufferUsageFlagBits::eTransferSrc, size);
		auto* storage  = temporary.get();
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, storage->Mapped().data(), size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
		storage->Flush(0, size);
		m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
		return {storage, 0};
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::ValidateFillRange(uint64_t vaddr, uint64_t size, bool is_gds) const {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	ValidateFillRange(vaddr, size, is_gds);
	if (is_gds) {
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	RecordKnownFill(vaddr, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) &&
	    !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::TryReadCleanBytes(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0 || size > UINT64_MAX - vaddr || m_gpu_modified_ranges.Intersects(vaddr, size)) {
		return false;
	}
	// An eager write-back takes its bytes out of the GPU-modified ranges when it is recorded and
	// writes them to guest memory only when it retires; until then exactly those bytes are stale.
	// Other bytes on the page are not touched by it.
	if (m_eager_pending_ranges.Intersects(vaddr, size)) {
		return false;
	}
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	return Libs::LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	// All downloads queued here share the 64 MiB staging buffer within one command buffer, so
	// keep a round's total below it. Dirty buffers that do not fit stay cached for a later round
	// (PPSA10595's stage load retired more than 64 MiB at once and exited).
	constexpr uint64_t DownloadBudget = 48ull * 1024 * 1024;
	uint64_t           download_bytes = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			if (buffer.Size() > DownloadBudget - download_bytes) {
				return false;
			}
			download_bytes += buffer.Size();
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
