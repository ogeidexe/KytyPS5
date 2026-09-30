#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_

// KYTY_GPU_WAIT_DIAGNOSTICS=1 only: a fixed-size ring of image layout barriers, view creations and
// rendering begins, dumped after a device loss so the history of the attachments the GPU stopped
// on can be reconstructed. Recording is a few stores; nothing is formatted until the dump.

#include "common/gpuWaitDiagnostics.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics::ImageHistory {

enum class Kind : uint32_t { Barrier, ViewCreated, BeginRendering };

struct Event {
	uint64_t    tick   = 0;
	Kind        kind   = Kind::Barrier;
	VkImage     image  = VK_NULL_HANDLE;
	VkImageView view   = VK_NULL_HANDLE;
	int32_t     old_layout = 0, new_layout = 0;
	uint32_t    level = 0, layer = 0, layer_count = 0, aspect = 0;
	uint64_t    src_access = 0, dst_access = 0;
	uint32_t    width = 0, height = 0, load_clear = 0;
	uint32_t    marker = 0; // GPU checkpoint counter when recorded; orders events against draws
};

// Set by the checkpoint code so events can be ordered against draw markers without a dependency
// cycle between the two headers.
inline std::atomic<uint32_t>* g_marker_counter = nullptr;

struct Ring {
	static constexpr uint32_t Size = 8192;
	std::array<Event, Size>   events {};
	std::atomic<uint64_t>     next {0};
};

inline Ring& GetRing() {
	static Ring ring;
	return ring;
}

inline void Record(const Event& event) {
	if (!Common::GpuWaitDiagnostics::Enabled()) {
		return;
	}
	auto& ring = GetRing();
	auto& slot = ring.events[ring.next.fetch_add(1, std::memory_order_relaxed) % Ring::Size];
	slot       = event;
	slot.marker = g_marker_counter != nullptr ? g_marker_counter->load(std::memory_order_relaxed) : 0;
}

// Prints the newest events that touch a depth/stencil aspect, plus rendering begins with a depth
// attachment, oldest first.
inline void Dump(uint32_t max_events = 400) {
	auto&          ring  = GetRing();
	const uint64_t end   = ring.next.load();
	const uint64_t begin = end > Ring::Size ? end - Ring::Size : 0;
	uint64_t       first = end;
	uint32_t       found = 0;
	for (uint64_t i = end; i > begin && found < max_events; i--) {
		const auto& e = ring.events[(i - 1) % Ring::Size];
		const bool  depth = (e.aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0 ||
		                   (e.kind == Kind::BeginRendering && e.view != VK_NULL_HANDLE);
		if (depth) {
			first = i - 1;
			found++;
		}
	}
	std::printf("[image-history] %u depth-related events of %" PRIu64 " recorded:\n", found, end);
	for (uint64_t i = first; i < end; i++) {
		const auto& e = ring.events[i % Ring::Size];
		switch (e.kind) {
			case Kind::Barrier:
				if ((e.aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) == 0) {
					continue;
				}
				std::printf("[image-history] m=%u tick %" PRIu64 " barrier image=%p level=%u layer=%u+%u"
				            " layout %d->%d access 0x%" PRIx64 "->0x%" PRIx64 "\n",
				            e.marker & 0x3ffffu, e.tick, static_cast<void*>(e.image), e.level,
				            e.layer, e.layer_count,
				            e.old_layout, e.new_layout, e.src_access, e.dst_access);
				break;
			case Kind::ViewCreated:
				if ((e.aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) == 0) {
					continue;
				}
				std::printf("[image-history] m=%u tick %" PRIu64 " view image=%p view=%p level=%u"
				            " layer=%u+%u\n",
				            e.marker & 0x3ffffu, e.tick, static_cast<void*>(e.image),
				            static_cast<void*>(e.view), e.level,
				            e.layer, e.layer_count);
				break;
			case Kind::BeginRendering:
				if (e.view == VK_NULL_HANDLE) {
					continue;
				}
				std::printf("[image-history] m=%u submit %" PRIu64 " begin rendering depth view=%p"
				            " layout=%d %s %ux%u layers=%u\n",
				            e.marker & 0x3ffffu, e.tick, static_cast<void*>(e.view), e.new_layout,
				            e.load_clear ? "clear" : "load", e.width, e.height, e.layer_count);
				break;
		}
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::ImageHistory

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_
