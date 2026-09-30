#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_GPUCHECKPOINTS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_GPUCHECKPOINTS_H_

// With KYTY_GPU_WAIT_DIAGNOSTICS=1 on a device exposing VK_NV_device_diagnostic_checkpoints, every
// draw and dispatch drops a checkpoint naming its debug record. After a device loss (a GPU hang
// the OS reset) the driver reports the last checkpoints the GPU reached, which identifies the
// work that hung it. Without the extension or the switch every call here is a no-op.

#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace Libs::Graphics::GpuCheckpoints {

struct Record {
	uint32_t op     = 0;
	uint64_t submit = 0;
	uint32_t arg0 = 0, arg1 = 0, arg2 = 0, arg3 = 0;
	uint64_t arg4 = 0;
};

struct State {
	PFN_vkCmdSetCheckpointNV       set_checkpoint = nullptr;
	PFN_vkGetQueueCheckpointDataNV get_data       = nullptr;
	static constexpr uint32_t      Size           = 1u << 18;
	std::unique_ptr<Record[]>      records;
	std::atomic<uint32_t>          next {1};
	std::atomic<bool>              reported {false};
	PFN_vkGetDeviceFaultInfoEXT    get_fault = nullptr;
	VkDevice                       device    = VK_NULL_HANDLE;
};

inline State& GetState() {
	static State state;
	return state;
}

// Call after VULKAN_HPP_DEFAULT_DISPATCHER was initialized for a device created with the extension.
inline void Initialize(vk::Device device) {
	auto& s          = GetState();
	s.device         = static_cast<VkDevice>(device);
	s.get_fault      = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT;
	if (s.get_fault != nullptr) {
		std::printf("[gpu-wait] device fault reporting enabled\n");
	}
	s.set_checkpoint = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV;
	s.get_data       = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV;
	if (s.set_checkpoint != nullptr && s.get_data != nullptr) {
		s.records = std::make_unique<Record[]>(State::Size);
		std::printf("[gpu-wait] device diagnostic checkpoints enabled\n");
	} else {
		s.set_checkpoint = nullptr;
	}
}

inline void Mark(vk::CommandBuffer command, const Record& record) {
	auto& s = GetState();
	if (s.set_checkpoint == nullptr || !command) {
		return;
	}
	auto index = s.next.fetch_add(1, std::memory_order_relaxed) & (State::Size - 1u);
	if (index == 0) {
		index = 1; // a null marker is indistinguishable from none
	}
	s.records[index] = record;
	s.set_checkpoint(static_cast<VkCommandBuffer>(command),
	                 reinterpret_cast<void*>(static_cast<uintptr_t>(index)));
}

// Valid only after the device reported VK_ERROR_DEVICE_LOST. Prints once.
inline void ReportDeviceFault(State& s) {
	if (s.get_fault == nullptr) {
		return;
	}
	VkDeviceFaultCountsEXT counts {};
	counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
	if (s.get_fault(s.device, &counts, nullptr) != VK_SUCCESS) {
		std::printf("[gpu-wait] device fault info unavailable\n");
		return;
	}
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendor(counts.vendorInfoCount);
	VkDeviceFaultInfoEXT                     info {};
	info.sType             = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
	info.pAddressInfos     = addresses.data();
	info.pVendorInfos      = vendor.data();
	counts.vendorBinarySize = 0; // not requested
	const auto result      = s.get_fault(s.device, &counts, &info);
	std::printf("[gpu-wait] device fault (%d): \"%s\", %u address record(s), %u vendor record(s)\n",
	            static_cast<int>(result), info.description, counts.addressInfoCount,
	            counts.vendorInfoCount);
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		// Types: 1 read-invalid, 2 write-invalid, 3 execute-invalid, 4 IP unknown, 5 IP invalid,
		// 6 IP fault.
		std::printf("[gpu-wait]   address type=%d reported=0x%016" PRIx64 " precision=0x%" PRIx64 "\n",
		            static_cast<int>(addresses[i].addressType),
		            static_cast<uint64_t>(addresses[i].reportedAddress),
		            static_cast<uint64_t>(addresses[i].addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		std::printf("[gpu-wait]   vendor \"%s\" code=0x%" PRIx64 " data=0x%" PRIx64 "\n",
		            vendor[i].description, static_cast<uint64_t>(vendor[i].vendorFaultCode),
		            static_cast<uint64_t>(vendor[i].vendorFaultData));
	}
	std::fflush(stdout);
}

inline void ReportDeviceLost(vk::Queue queue) {
	auto& s = GetState();
	if (s.reported.exchange(true)) {
		return;
	}
	ReportDeviceFault(s);
	if (s.get_data == nullptr) {
		return;
	}
	uint32_t count = 0;
	s.get_data(static_cast<VkQueue>(queue), &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count);
	for (auto& d: data) {
		d.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
		d.pNext = nullptr;
	}
	s.get_data(static_cast<VkQueue>(queue), &count, data.data());
	std::printf("[gpu-wait] device lost: %u checkpoint(s) reached, newest marker %u\n", count,
	            (s.next.load() - 1u) & (State::Size - 1u));
	for (uint32_t i = 0; i < count; i++) {
		const auto index = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker));
		const auto& r    = s.records[index & (State::Size - 1u)];
		std::printf("[gpu-wait]   stage=0x%08x marker=%u op=%u submit=%" PRIu64
		            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		            static_cast<unsigned>(data[i].stage), index, r.op, r.submit, r.arg0, r.arg1,
		            r.arg2, r.arg3, r.arg4);
		// The work recorded right after a reached marker; after the bottom-of-pipe marker this is
		// what the GPU never finished. Op 0xD0 is a draw about to be emitted: args are vertex
		// program, pixel program, mesh groups (0 = not mesh), instances, index count.
		for (uint32_t k = 1; k <= 4; k++) {
			const auto& n = s.records[(index + k) & (State::Size - 1u)];
			std::printf("[gpu-wait]       +%u op=%u submit=%" PRIu64
			            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
			            k, n.op, n.submit, n.arg0, n.arg1, n.arg2, n.arg3, n.arg4);
		}
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuCheckpoints

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_GPUCHECKPOINTS_H_
