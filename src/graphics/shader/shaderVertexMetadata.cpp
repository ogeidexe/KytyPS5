#include "graphics/shader/shaderVertexMetadata.h"

#include "common/frameStats.h"

#include <array>
#include <cstring>

namespace Libs::Graphics {

namespace {
ShaderCleanGuestReader g_clean_reader         = nullptr;
void*                  g_clean_reader_context = nullptr;
} // namespace

void ShaderSetCleanGuestReader(ShaderCleanGuestReader reader, void* context) {
	g_clean_reader         = reader;
	g_clean_reader_context = context;
}

bool ShaderTryReadGuestClean(void* dst, const void* src, size_t size) {
	return g_clean_reader != nullptr &&
	       g_clean_reader(g_clean_reader_context, reinterpret_cast<uint64_t>(src), dst, size);
}

void ShaderReadGuest(void* dst, const void* src, size_t size) {
	if (size == 0) {
		return;
	}
	if (ShaderTryReadGuestClean(dst, src, size)) {
		Common::FrameStats::g_guest_reads_clean.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	Common::FrameStats::g_guest_reads_direct.fetch_add(1, std::memory_order_relaxed);
	std::memcpy(dst, src, size);
}

bool ShaderReadVertexMetadata(const ShaderMappedData& data, uint32_t max_user_sgprs,
                              ShaderVertexMetadata& metadata, std::string* error) {
	if (data.user_data == nullptr) {
		return ShaderError::Fail(error, "missing AGC user-data header");
	}

	ShaderUserData user_data {};
	ShaderReadGuest(&user_data, data.user_data, sizeof(user_data));

	constexpr uint32_t DirectResourceCount =
	    static_cast<uint32_t>(AgcDirectResourceType::Last) + 1u;
	if (user_data.direct_resource_count > DirectResourceCount) {
		return ShaderError::Fail(error,
		                         "AGC direct-resource count exceeds the known resource domain");
	}

	std::array<uint16_t, DirectResourceCount> direct_offsets {};
	const auto                                direct_size =
	    static_cast<uint64_t>(user_data.direct_resource_count) * sizeof(uint16_t);
	if (direct_size != 0) {
		if (user_data.direct_resource_offset == nullptr) {
			return ShaderError::Fail(error, "missing AGC direct-resource offsets");
		}
		ShaderReadGuest(direct_offsets.data(), user_data.direct_resource_offset, direct_size);
	}

	ShaderVertexMetadata next;
	for (uint32_t type = 0; type < user_data.direct_resource_count; type++) {
		const auto reg = direct_offsets[type];
		if (reg == AGC_ILLEGAL_DIRECT_OFFSET) {
			continue;
		}
		switch (static_cast<AgcDirectResourceType>(type)) {
			case AgcDirectResourceType::PtrVertexBufferTable: next.vertex_buffer_reg = reg; break;
			case AgcDirectResourceType::PtrVertexAttribDescTable:
				next.vertex_attrib_reg = reg;
				break;
			default: break;
		}
	}

	if (next.vertex_attrib_reg >= 0 && next.vertex_buffer_reg < 0) {
		return ShaderError::Fail(error, "vertex attribute table requires a vertex buffer table");
	}
	if (next.vertex_buffer_reg < 0) {
		metadata = next;
		return true;
	}
	if (static_cast<uint32_t>(next.vertex_buffer_reg) + 1u >= max_user_sgprs) {
		return ShaderError::Fail(error, "vertex table pointer exceeds the user-SGPR domain");
	}
	if (next.vertex_attrib_reg < 0) {
		metadata = next;
		return true;
	}
	if (static_cast<uint32_t>(next.vertex_attrib_reg) + 1u >= max_user_sgprs) {
		return ShaderError::Fail(error, "vertex table pointer exceeds the user-SGPR domain");
	}
	if (data.num_input_semantics == 0 ||
	    data.num_input_semantics > ShaderVertexInputInfo::RES_MAX) {
		return ShaderError::Fail(error, "vertex semantic count is outside the supported domain");
	}

	if (data.input_semantics == nullptr) {
		return ShaderError::Fail(error, "missing vertex input semantics");
	}
	next.input_semantics = {data.input_semantics, data.num_input_semantics};
	metadata                   = next;
	return true;
}

} // namespace Libs::Graphics
