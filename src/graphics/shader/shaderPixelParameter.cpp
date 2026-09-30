#include "graphics/shader/shader.h"

#include "common/assert.h"

namespace Libs::Graphics {

namespace {

constexpr uint32_t PsInputOffsetMask = 0x0000001fu;
constexpr uint32_t PsInputFlatShade  = 0x00000400u;

} // namespace

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num ? info.interpolator_settings[input] & PsInputOffsetMask : input;
}

uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input) {
	std::array<bool, 32> used_locations {};
	for (const auto active_input: active_inputs) {
		used_locations[ShaderPixelParameterMappedLocation(info, active_input)] = true;
	}
	std::array<uint32_t, 64> group_locations;
	group_locations.fill(UINT32_MAX);
	for (const auto active_input: active_inputs) {
		const auto mapped = ShaderPixelParameterMappedLocation(info, active_input);
		const auto group  = mapped * 2u + ShaderPixelParameterIsFlat(info, active_input);
		auto&      location = group_locations[group];
		if (location == UINT32_MAX) {
			location = mapped;
			// Smooth and custom interpolation read the same vertex output. Only
			// differing flat/smooth rectangle outputs need separate locations.
			if (group_locations[group ^ 1u] != UINT32_MAX) {
				location = 0;
				while (location < used_locations.size() && used_locations[location]) {
					location++;
				}
				EXIT_NOT_IMPLEMENTED(location >= used_locations.size());
			}
			used_locations[location] = true;
		}

		if (active_input == input) {
			return location;
		}
	}
	return ShaderPixelParameterMappedLocation(info, input);
}

bool ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num && (info.interpolator_settings[input] & PsInputFlatShade) != 0 &&
	       !ShaderPixelParameterIsCustom(info, input);
}

bool ShaderPixelParameterUsesDefault(const ShaderPixelInputInfo& info, uint32_t input) {
	if (input >= info.input_num) {
		return false;
	}
	constexpr uint32_t PsInputUseDefault = 0x00000020u;
	const auto         settings          = info.interpolator_settings[input];
	return (settings & PsInputUseDefault) != 0 ||
	       (settings & PsInputOffsetMask) >= info.vs_export_count;
}

uint32_t ShaderPixelParameterDefaultBits(const ShaderPixelInputInfo& info, uint32_t input,
                                         uint32_t component) {
	constexpr uint32_t One = 0x3f800000u;
	const auto default_val = input < info.input_num ? (info.interpolator_settings[input] >> 8u) & 3u
	                                                : 0u;
	switch (default_val) {
		case 1: return component == 3 ? One : 0u;
		case 2: return component == 3 ? 0u : One;
		case 3: return One;
		default: return 0u;
	}
}

bool ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < 32u && (info.custom_interpolation_mask & (1u << input)) != 0;
}

} // namespace Libs::Graphics
