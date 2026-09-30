#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>

namespace Libs::Graphics {

namespace {

// Images the texture cache destroys are frequently recreated with the same parameters a few
// frames later (hundreds per second in some titles), and each one costs a dedicated device
// allocation and free in the kernel driver. A destroyed image is only returned here once the GPU
// is done with it (Image destruction is deferred until its tick completes), so an idle image with
// identical create parameters can be handed back instead. Its contents are undefined and its
// tracked layout is reset to UNDEFINED, exactly as for a freshly created image.
// KYTY_IMAGE_POOL=0 disables reuse.
struct ImagePoolKey {
	vk::ImageCreateFlags  flags;
	vk::ImageType         type = vk::ImageType::e2D;
	vk::Extent3D          extent;
	uint32_t              mips = 0, layers = 0;
	vk::Format            format = vk::Format::eUndefined;
	vk::ImageUsageFlags   usage;
	vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

	bool operator==(const ImagePoolKey&) const = default;
};

struct PooledImage {
	ImagePoolKey  key;
	vk::Image     image      = nullptr;
	VmaAllocation allocation = nullptr;
	uint64_t      bytes      = 0;
};

class ImagePool {
public:
	static constexpr uint64_t MaxBytes  = 256ull * 1024 * 1024;
	static constexpr size_t   MaxPerKey = 8;

	static bool Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_IMAGE_POOL");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}();
		return enabled;
	}

	bool Take(const ImagePoolKey& key, vk::Image& image, VmaAllocation& allocation) {
		std::lock_guard lock(m_mutex);
		for (auto it = m_entries.rbegin(); it != m_entries.rend(); ++it) {
			if (it->key == key) {
				image      = it->image;
				allocation = it->allocation;
				m_bytes -= it->bytes;
				m_entries.erase(std::next(it).base());
				return true;
			}
		}
		return false;
	}

	// Returns false when the image should be destroyed instead.
	bool Put(VmaAllocator allocator, const ImagePoolKey& key, vk::Image image,
	         VmaAllocation allocation) {
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, allocation, &info);
		const auto bytes = static_cast<uint64_t>(info.size);
		if (bytes > MaxBytes / 4) {
			return false;
		}
		std::vector<PooledImage> evicted;
		{
			std::lock_guard lock(m_mutex);
			const auto same = std::count_if(m_entries.begin(), m_entries.end(),
			                                [&](const PooledImage& e) { return e.key == key; });
			if (static_cast<size_t>(same) >= MaxPerKey) {
				return false;
			}
			m_entries.push_back({key, image, allocation, bytes});
			m_bytes += bytes;
			while (m_bytes > MaxBytes && !m_entries.empty()) {
				evicted.push_back(m_entries.front());
				m_bytes -= m_entries.front().bytes;
				m_entries.pop_front();
			}
		}
		for (const auto& e: evicted) {
			vmaDestroyImage(allocator, e.image, e.allocation);
		}
		return true;
	}

	void Flush(VmaAllocator allocator) {
		std::lock_guard lock(m_mutex);
		for (const auto& e: m_entries) {
			vmaDestroyImage(allocator, e.image, e.allocation);
		}
		m_entries.clear();
		m_bytes = 0;
	}

private:
	std::mutex              m_mutex;
	std::deque<PooledImage> m_entries;
	uint64_t                m_bytes = 0;
};

ImagePool& GetImagePool() {
	static ImagePool pool;
	return pool;
}

ImagePoolKey MakeImagePoolKey(const vk::ImageCreateInfo& info) {
	return {info.flags,       info.imageType, info.extent, info.mipLevels,
	        info.arrayLayers, info.format,    info.usage,  info.samples};
}

} // namespace

// KYTY_BDA_CAPTURE_REPLAY: memory VMA allocates for device-address use also gets the capture-replay
// flag, which a buffer created with the capture-replay flag must be bound to. The flags struct is
// VMA's own local, so updating it in place is allowed.
static PFN_vkAllocateMemory g_allocate_memory = nullptr;

static VkResult VKAPI_CALL AllocateMemoryCaptureReplay(VkDevice device,
                                                       const VkMemoryAllocateInfo* info,
                                                       const VkAllocationCallbacks* callbacks,
                                                       VkDeviceMemory* memory) {
	for (auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next != nullptr;
	     next = next->pNext) {
		if (next->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) {
			auto* flags = const_cast<VkMemoryAllocateFlagsInfo*>(
			    reinterpret_cast<const VkMemoryAllocateFlagsInfo*>(next));
			if ((flags->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) != 0) {
				flags->flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
			}
		}
	}
	return g_allocate_memory(device, info, callbacks, memory);
}

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	if (bda_capture_replay) {
		g_allocate_memory          = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory;
		functions.vkAllocateMemory = AllocateMemoryCaptureReplay;
	}

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	GetImagePool().Flush(allocator);
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	const bool pooled = ImagePool::Enabled() && image_info.pNext == nullptr &&
	                    image_info.initialLayout == vk::ImageLayout::eUndefined &&
	                    GetImagePool().Take(MakeImagePoolKey(image_info), image.image,
	                                        image.allocation);
	if (!pooled) {
		VmaAllocationCreateInfo alloc_info {};
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

		vk::Image::CType native_image = VK_NULL_HANDLE;
		const auto        result       = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		image.image = native_image;
		if (result != vk::Result::eSuccess) {
			LogMemoryBudget();
			return false;
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vk::ImageCreateInfo info {};
	info.flags       = image.flags;
	info.imageType   = image.image_type;
	info.extent      = image.extent;
	info.mipLevels   = image.mip_levels;
	info.arrayLayers = image.layers;
	info.format      = image.format;
	info.usage       = image.usage;
	info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
	if (!ImagePool::Enabled() ||
	    !GetImagePool().Put(allocator, MakeImagePoolKey(info), image.image, image.allocation)) {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
