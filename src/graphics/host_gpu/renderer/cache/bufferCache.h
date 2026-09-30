#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// Rejects a fill range that is misaligned or out of bounds. Callers that split one guest
	// write into several fills validate the whole range up front so a rejected tail cannot
	// leave a partially written value behind.
	void ValidateFillRange(uint64_t vaddr, uint64_t size, bool is_gds) const;
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Copies [vaddr, vaddr + size) from guest memory when guest memory is authoritative for exactly
	// those bytes: none is GPU-modified or awaiting an eager write-back, and no GPU-modified image
	// overlaps them. Otherwise returns false and reads nothing. GPU thread only. Unlike a direct
	// load this never faults, so bytes sharing a page with GPU-written data read without a drain.
	[[nodiscard]] bool TryReadCleanBytes(uint64_t vaddr, void* data, uint64_t size);
	// KYTY_GPU_HAZARD_VERIFY=1 only: reports a buffer handle about to be bound by the command
	// buffer being recorded after the cache deleted it in an earlier tick. Such a buffer is freed
	// once that earlier tick completes, possibly before this command buffer executes.
	void VerifyBindingAlive(vk::Buffer handle, const char* use, uint64_t submit_id);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// True when [vaddr, vaddr + size) was last written, on the GPU, by one fill with *value.
	// Lets callers use GPU-resident fill results without draining the GPU to read them back.
	[[nodiscard]] bool TryGetKnownFill(uint64_t vaddr, uint64_t size, uint32_t* value);
	// Records that the GPU work recorded last filled [vaddr, vaddr + size) with value.
	void RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value);
	// Guest threads that fault on GPU-written memory mark its page hot. Before each submit, pending
	// GPU writes to hot pages are copied out in the same command buffer and written back on
	// completion, after which the page is unprotected: the next guest read needs no fault, no
	// GPU-thread round trip and no wait. Runs on the GPU thread.
	void               WriteBackHotPages();
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	// KYTY_BUFFER_VERIFY=1 only: checks that the GPU copy of a range about to be read matches
	// guest memory on every page neither side has modified since the last synchronization.
	void VerifyCoherence(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool CollectDownloadCopies(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                         std::vector<vk::BufferCopy>& copies,
	                                         uint64_t&                    total_size);
	// Reads back GPU writes that were already submitted on a separate command buffer, so the
	// CPU waits only for that work instead of flushing and draining the one being recorded.
	[[nodiscard]] bool ReadbackSubmitted(Buffer& buffer, uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	vk::CommandPool   m_readback_pool      = nullptr;
	vk::CommandBuffer m_readback_command   = nullptr;
	vk::Semaphore     m_readback_semaphore = nullptr;
	uint64_t          m_readback_tick      = 0;
	// Hot page (4 KiB aligned) -> tick of its in-flight eager write-back, 0 when none.
	std::map<uint64_t, uint64_t> m_hot_pages;
	// Bytes an in-flight eager write-back took out of the GPU-modified ranges and has not yet
	// written to guest memory. Cleared per page when the page's write-back retires.
	RangeSet m_eager_pending_ranges;
	struct RetiredBuffer {
		uint64_t tick = 0, guest = 0, size = 0;
	};
	// KYTY_GPU_HAZARD_VERIFY only: deleted buffers whose destruction is still deferred.
	std::unordered_map<VkBuffer, RetiredBuffer> m_retired_buffers;
	void RetireHotPages(uint64_t vaddr, uint64_t size, bool wait);
	bool VerifyWrittenBack(uint64_t page);
	bool EagerDownload(Buffer& buffer, uint64_t vaddr, uint64_t size);

	struct KnownFill {
		uint64_t end;
		uint32_t value;
	};
	void                                 ForgetKnownFills(uint64_t vaddr, uint64_t size);
	std::mutex                           m_known_fills_mutex;
	std::map<uint64_t, KnownFill>        m_known_fills;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
