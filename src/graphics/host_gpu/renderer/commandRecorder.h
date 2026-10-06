#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "common/common.h"
#include "common/frameStats.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Vulkan commands recorded on the GPU thread for the recording thread to replay into a command
// buffer. Every command and every array it points to lives in the stream's chunks until the
// stream is cleared, so recorded commands never reference caller memory.
class CommandStream {
public:
	CommandStream() = default;
	~CommandStream() { Clear(); }
	KYTY_CLASS_NO_COPY(CommandStream);

	template <typename F>
	void Record(F&& function) {
		using T = std::decay_t<F>;
		static_assert(alignof(T) <= PayloadAlign);
		auto* header = static_cast<Header*>(Allocate(HeaderSize + sizeof(T), PayloadAlign));
		new (Payload(header)) T(std::forward<F>(function));
		header->execute = [](void* payload, vk::CommandBuffer command) {
			(*static_cast<T*>(payload))(command);
		};
		header->destroy = std::is_trivially_destructible_v<T>
		                      ? nullptr
		                      : +[](void* payload) { static_cast<T*>(payload)->~T(); };
		header->next = nullptr;
		if (m_last != nullptr) {
			m_last->next = header;
		} else {
			m_first = header;
		}
		m_last = header;
		m_count++;
	}

	// A copy of count elements that stays valid until the stream is cleared.
	template <typename T>
	const T* Persist(const T* data, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (data == nullptr || count == 0) {
			return data;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T) < 8 ? 8 : alignof(T)));
		std::memcpy(copy, data, sizeof(T) * count);
		return copy;
	}

	// One thread records and publishes while another executes what has been published, so the
	// command buffer is replayed while it is still being recorded.

	// Recording thread: makes every command recorded so far visible to ExecutePublished.
	void Publish() noexcept {
		m_published.store(m_last, std::memory_order_release);
		m_published_count = m_count;
	}
	[[nodiscard]] size_t UnpublishedCount() const noexcept { return m_count - m_published_count; }

	// Executing thread: true when published commands have not been executed yet.
	[[nodiscard]] bool HasPublishedWork() const noexcept {
		return m_published.load(std::memory_order_acquire) != m_executed;
	}
	// Executing thread: replays the commands published since the last call, in recording order.
	void ExecutePublished(vk::CommandBuffer command) {
		auto* last = m_published.load(std::memory_order_acquire);
		if (last == nullptr || last == m_executed) {
			return;
		}
		// The recording thread can still be linking a new command to last: never read last->next.
		for (auto* header = m_executed != nullptr ? m_executed->next : m_first;;
		     header       = header->next) {
			header->execute(Payload(header), command);
			if (header == last) {
				break;
			}
		}
		m_executed = last;
	}

	// Only once the recording thread has stopped using the stream.
	void Clear() {
		for (auto* header = m_first; header != nullptr; header = header->next) {
			if (header->destroy != nullptr) {
				header->destroy(Payload(header));
			}
		}
		m_first           = nullptr;
		m_last            = nullptr;
		m_count           = 0;
		m_chunk           = 0;
		m_offset          = 0;
		m_published_count = 0;
		m_executed        = nullptr;
		m_published.store(nullptr, std::memory_order_relaxed);
		m_large.clear();
	}

	[[nodiscard]] bool   Empty() const noexcept { return m_first == nullptr; }
	[[nodiscard]] size_t Count() const noexcept { return m_count; }

private:
	struct Header {
		void (*execute)(void*, vk::CommandBuffer);
		void (*destroy)(void*);
		Header* next;
	};
	static constexpr size_t PayloadAlign = 16;
	static constexpr size_t HeaderSize   = (sizeof(Header) + PayloadAlign - 1) & ~(PayloadAlign - 1);
	static constexpr size_t ChunkSize    = 64 * 1024;

	static void* Payload(Header* header) { return reinterpret_cast<std::byte*>(header) + HeaderSize; }

	void* Allocate(size_t size, size_t align) {
		if (size > ChunkSize / 2) {
			// Rare oversized arrays get their own block, released when the stream is cleared.
			m_large.emplace_back(new (std::align_val_t {PayloadAlign}) std::byte[size]);
			return m_large.back().get();
		}
		auto offset = (m_offset + align - 1) & ~(align - 1);
		if (m_chunk >= m_chunks.size() || offset + size > ChunkSize) {
			if (m_chunk < m_chunks.size() && offset + size > ChunkSize) {
				m_chunk++;
			}
			if (m_chunk >= m_chunks.size()) {
				m_chunks.emplace_back(new (std::align_val_t {PayloadAlign}) std::byte[ChunkSize]);
			}
			offset = 0;
		}
		m_offset = offset + size;
		return m_chunks[m_chunk].get() + offset;
	}

	struct AlignedDelete {
		void operator()(std::byte* data) const {
			::operator delete[](data, std::align_val_t {PayloadAlign});
		}
	};

	std::vector<std::unique_ptr<std::byte[], AlignedDelete>> m_chunks;
	std::vector<std::unique_ptr<std::byte[], AlignedDelete>> m_large;
	size_t                                                   m_chunk  = 0;
	size_t                                                   m_offset = 0;
	Header*                                                  m_first  = nullptr;
	Header*                                                  m_last   = nullptr;
	size_t                                                   m_count  = 0;
	size_t                                                   m_published_count = 0;
	std::atomic<Header*>                                     m_published {nullptr};
	Header*                                                  m_executed = nullptr; // executing thread
};

// Commands that read or write memory (draws, dispatches, copies, clears, render pass begins)
// recorded by this thread. A barrier with none recorded since the previous one orders nothing new.
inline thread_local uint64_t g_recorded_work = 0;

// What renderer code records commands through. Immediate recorders forward to a Vulkan command
// buffer at once; deferred ones copy the call (and every array it references) into a stream for
// the recording thread. The method names and forms follow vk::CommandBuffer.
class CommandRecorder {
public:
	CommandRecorder() = default;
	explicit CommandRecorder(vk::CommandBuffer immediate): m_immediate(immediate) {}
	explicit CommandRecorder(CommandStream* stream): m_stream(stream) {}

	explicit operator bool() const noexcept { return m_stream != nullptr || m_immediate; }
	[[nodiscard]] bool Deferred() const noexcept { return m_stream != nullptr; }

	// Runs function with the Vulkan command buffer now, or when the stream is replayed. It must
	// own everything it captures (copies, or memory from Persist).
	template <typename F>
	void Run(F&& function) const {
		if (m_stream != nullptr) {
			m_stream->Record(std::forward<F>(function));
		} else {
			function(m_immediate);
		}
	}

	template <typename T>
	const T* Persist(const T* data, size_t count) const {
		return m_stream != nullptr ? m_stream->Persist(data, count) : data;
	}

	void bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) const {
		Run([=](vk::CommandBuffer c) { c.bindPipeline(point, pipeline); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t first,
	                        uint32_t count, const vk::DescriptorSet* sets, uint32_t offset_count,
	                        const uint32_t* offsets) const {
		const auto* s = Persist(sets, count);
		const auto* o = Persist(offsets, offset_count);
		Run([=](vk::CommandBuffer c) {
			c.bindDescriptorSets(point, layout, first, count, s, offset_count, o);
		});
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		const auto* v = Persist(static_cast<const std::byte*>(values), size);
		Run([=](vk::CommandBuffer c) { c.pushConstants(layout, stages, offset, size, v); });
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSetKHR(point, layout, set, static_cast<uint32_t>(writes.size()), writes.data());
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          uint32_t count, const vk::WriteDescriptorSet* writes) const {
		const auto* copied = Persist(writes, count);
		if (m_stream != nullptr) {
			// The writes point at descriptor info arrays: persist those too.
			auto* w = const_cast<vk::WriteDescriptorSet*>(copied);
			for (uint32_t i = 0; i < count; i++) {
				w[i].pImageInfo       = Persist(w[i].pImageInfo, w[i].pImageInfo ? w[i].descriptorCount : 0);
				w[i].pBufferInfo      = Persist(w[i].pBufferInfo, w[i].pBufferInfo ? w[i].descriptorCount : 0);
				w[i].pTexelBufferView = Persist(w[i].pTexelBufferView,
				                                w[i].pTexelBufferView ? w[i].descriptorCount : 0);
				EXIT_IF(w[i].pNext != nullptr);
			}
		}
		Run([=](vk::CommandBuffer c) { c.pushDescriptorSetKHR(point, layout, set, count, copied); });
	}

	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		Common::FrameStats::g_barriers.fetch_add(1, std::memory_order_relaxed);
		const auto* m = Persist(memory, memory_count);
		const auto* b = Persist(buffers, buffer_count);
		const auto* i = Persist(images, image_count);
		Run([=](vk::CommandBuffer c) {
			c.pipelineBarrier(src, dst, flags, memory_count, m, buffer_count, b, image_count, i);
		});
	}
	void pipelineBarrier2(const vk::DependencyInfo& info) const {
		Common::FrameStats::g_barriers.fetch_add(1, std::memory_order_relaxed);
		if (m_stream == nullptr) {
			m_immediate.pipelineBarrier2(info);
			return;
		}
		EXIT_IF(info.pNext != nullptr);
		vk::DependencyInfo copy  = info;
		copy.pMemoryBarriers       = Persist(info.pMemoryBarriers, info.memoryBarrierCount);
		copy.pBufferMemoryBarriers = Persist(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount);
		copy.pImageMemoryBarriers  = Persist(info.pImageMemoryBarriers, info.imageMemoryBarrierCount);
		Run([=](vk::CommandBuffer c) { c.pipelineBarrier2(copy); });
	}

	void beginRendering(const vk::RenderingInfo* info) const {
		++g_recorded_work;
		if (m_stream == nullptr) {
			m_immediate.beginRendering(info);
			return;
		}
		EXIT_IF(info->pNext != nullptr);
		vk::RenderingInfo copy = *info;
		copy.pColorAttachments  = Persist(info->pColorAttachments, info->colorAttachmentCount);
		copy.pDepthAttachment   = Persist(info->pDepthAttachment, info->pDepthAttachment ? 1u : 0u);
		copy.pStencilAttachment = Persist(info->pStencilAttachment, info->pStencilAttachment ? 1u : 0u);
		for (uint32_t k = 0; k < copy.colorAttachmentCount; k++) {
			EXIT_IF(copy.pColorAttachments[k].pNext != nullptr);
		}
		Run([=](vk::CommandBuffer c) { c.beginRendering(&copy); });
	}
	void beginRendering(const vk::RenderingInfo& info) const { beginRendering(&info); }
	void endRendering() const {
		Run([](vk::CommandBuffer c) { c.endRendering(); });
	}

	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex,
	          uint32_t first_instance) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.draw(vertices, instances, first_vertex, first_instance); });
	}
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		Run([=](vk::CommandBuffer c) {
		++g_recorded_work;
			c.drawIndexed(indices, instances, first_index, vertex_offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.drawIndirect(buffer, offset, count, stride); });
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.drawIndexedIndirect(buffer, offset, count, stride); });
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.drawMeshTasksEXT(x, y, z); });
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.dispatchIndirect(buffer, offset); });
	}

	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		const auto* b  = Persist(buffers, count);
		const auto* o  = Persist(offsets, count);
		const auto* s  = Persist(sizes, count);
		const auto* st = Persist(strides, count);
		Run([=](vk::CommandBuffer c) { c.bindVertexBuffers2(first, count, b, o, s, st); });
	}
	void bindVertexBuffers(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                       const vk::DeviceSize* offsets) const {
		const auto* b = Persist(buffers, count);
		const auto* o = Persist(offsets, count);
		Run([=](vk::CommandBuffer c) { c.bindVertexBuffers(first, count, b, o); });
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Run([=](vk::CommandBuffer c) { c.bindIndexBuffer(buffer, offset, type); });
	}

	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
		const auto* v = Persist(viewports, count);
		Run([=](vk::CommandBuffer c) { c.setViewport(first, count, v); });
	}
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
		const auto* s = Persist(scissors, count);
		Run([=](vk::CommandBuffer c) { c.setScissor(first, count, s); });
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		const auto* v = Persist(viewports, count);
		Run([=](vk::CommandBuffer c) { c.setViewportWithCount(count, v); });
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		const auto* s = Persist(scissors, count);
		Run([=](vk::CommandBuffer c) { c.setScissorWithCount(count, s); });
	}
	void setLineWidth(float width) const {
		Run([=](vk::CommandBuffer c) { c.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		const auto* k = Persist(constants, 4);
		Run([=](vk::CommandBuffer c) { c.setBlendConstants(k); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		Run([=](vk::CommandBuffer c) { c.setDepthBias(constant, clamp, slope); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer c) { c.setDepthBiasEnable(enable); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		Run([=](vk::CommandBuffer c) { c.setDepthBounds(min_bounds, max_bounds); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer c) { c.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer c) { c.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		Run([=](vk::CommandBuffer c) { c.setDepthCompareOp(op); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer c) { c.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		Run([=](vk::CommandBuffer c) { c.setStencilOp(faces, fail, pass, depth_fail, compare); });
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Run([=](vk::CommandBuffer c) { c.setStencilCompareMask(faces, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Run([=](vk::CommandBuffer c) { c.setStencilWriteMask(faces, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		Run([=](vk::CommandBuffer c) { c.setStencilReference(faces, reference); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		const auto* e = Persist(enables, count);
		Run([=](vk::CommandBuffer c) { c.setColorWriteEnableEXT(count, e); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		Run([=](vk::CommandBuffer c) { c.setAttachmentFeedbackLoopEnableEXT(aspects); });
	}

	void copyBuffer(vk::Buffer src, vk::Buffer dst, uint32_t count,
	                const vk::BufferCopy* regions) const {
		++g_recorded_work;
		const auto* r = Persist(regions, count);
		Run([=](vk::CommandBuffer c) { c.copyBuffer(src, dst, count, r); });
	}
	void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
	               vk::ImageLayout dst_layout, uint32_t count, const vk::ImageCopy* regions) const {
		++g_recorded_work;
		const auto* r = Persist(regions, count);
		Run([=](vk::CommandBuffer c) { c.copyImage(src, src_layout, dst, dst_layout, count, r); });
	}
	void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout, uint32_t count,
	                       const vk::BufferImageCopy* regions) const {
		++g_recorded_work;
		const auto* r = Persist(regions, count);
		Run([=](vk::CommandBuffer c) { c.copyBufferToImage(src, dst, layout, count, r); });
	}
	void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst, uint32_t count,
	                       const vk::BufferImageCopy* regions) const {
		++g_recorded_work;
		const auto* r = Persist(regions, count);
		Run([=](vk::CommandBuffer c) { c.copyImageToBuffer(src, layout, dst, count, r); });
	}
	void blitImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
	               vk::ImageLayout dst_layout, uint32_t count, const vk::ImageBlit* regions,
	               vk::Filter filter) const {
		++g_recorded_work;
		const auto* r = Persist(regions, count);
		Run([=](vk::CommandBuffer c) {
			c.blitImage(src, src_layout, dst, dst_layout, count, r, filter);
		});
	}
	void copyBuffer(vk::Buffer src, vk::Buffer dst,
	                vk::ArrayProxy<const vk::BufferCopy> const& regions) const {
		copyBuffer(src, dst, static_cast<uint32_t>(regions.size()), regions.data());
	}
	void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
	               vk::ImageLayout dst_layout, vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
		copyImage(src, src_layout, dst, dst_layout, static_cast<uint32_t>(regions.size()),
		          regions.data());
	}
	void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyBufferToImage(src, dst, layout, static_cast<uint32_t>(regions.size()), regions.data());
	}
	void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyImageToBuffer(src, layout, dst, static_cast<uint32_t>(regions.size()), regions.data());
	}
	void blitImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
	               vk::ImageLayout dst_layout, vk::ArrayProxy<const vk::ImageBlit> const& regions,
	               vk::Filter filter) const {
		blitImage(src, src_layout, dst, dst_layout, static_cast<uint32_t>(regions.size()),
		          regions.data(), filter);
	}
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags,
	                     vk::ArrayProxy<const vk::MemoryBarrier> const&       memory,
	                     vk::ArrayProxy<const vk::BufferMemoryBarrier> const& buffers,
	                     vk::ArrayProxy<const vk::ImageMemoryBarrier> const&  images) const {
		pipelineBarrier(src, dst, flags, static_cast<uint32_t>(memory.size()), memory.data(),
		                static_cast<uint32_t>(buffers.size()), buffers.data(),
		                static_cast<uint32_t>(images.size()), images.data());
	}
	void setViewport(uint32_t first, vk::ArrayProxy<const vk::Viewport> const& viewports) const {
		setViewport(first, static_cast<uint32_t>(viewports.size()), viewports.data());
	}
	void setScissor(uint32_t first, vk::ArrayProxy<const vk::Rect2D> const& scissors) const {
		setScissor(first, static_cast<uint32_t>(scissors.size()), scissors.data());
	}
	void resolveImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
	                  vk::ImageLayout dst_layout,
	                  vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
		++g_recorded_work;
		const auto  count = static_cast<uint32_t>(regions.size());
		const auto* r     = Persist(regions.data(), count);
		Run([=](vk::CommandBuffer c) {
			c.resolveImage(src, src_layout, dst, dst_layout, count, r);
		});
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) const {
		++g_recorded_work;
		const auto* k = Persist(color, 1);
		const auto* r = Persist(ranges, count);
		Run([=](vk::CommandBuffer c) { c.clearColorImage(image, layout, k, count, r); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
	                     vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
		clearColorImage(image, layout, &color, ranges.size(), ranges.data());
	}
	void clearAttachments(uint32_t attachment_count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const {
		++g_recorded_work;
		const auto* a = Persist(attachments, attachment_count);
		const auto* r = Persist(rects, rect_count);
		Run([=](vk::CommandBuffer c) { c.clearAttachments(attachment_count, a, rect_count, r); });
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) const {
		++g_recorded_work;
		const auto* k = Persist(value, 1);
		const auto* r = Persist(ranges, count);
		Run([=](vk::CommandBuffer c) { c.clearDepthStencilImage(image, layout, k, count, r); });
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t data) const {
		++g_recorded_work;
		Run([=](vk::CommandBuffer c) { c.fillBuffer(buffer, offset, size, data); });
	}
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const {
		++g_recorded_work;
		const auto* d = Persist(static_cast<const std::byte*>(data), static_cast<size_t>(size));
		Run([=](vk::CommandBuffer c) { c.updateBuffer(buffer, offset, size, d); });
	}

	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
		Run([=](vk::CommandBuffer c) { c.resetQueryPool(pool, first, count); });
	}
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		Run([=](vk::CommandBuffer c) { c.beginQuery(pool, query, flags); });
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer c) { c.endQuery(pool, query); });
	}

private:
	vk::CommandBuffer m_immediate;
	CommandStream*    m_stream = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
