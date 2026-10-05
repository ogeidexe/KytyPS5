#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "common/frameStats.h"
#include "common/gpuWaitDiagnostics.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
}

bool ValidateShaderGuestMemoryRange(void*, uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::TryClampRangeSize(address, size) != 0;
}

// Dependency trace of one shader resource (SRT) evaluation. MaterializeResources is a pure function
// of the user data, the shader base and the guest memory it reads through three channels: direct
// word reads, strict (GPU-clean) reads and range validation. The trace records every such access
// with its result. If a later draw of the same program has identical user data and shader base and
// every recorded access still returns the same result, the evaluation would repeat exactly, so the
// outputs left in the cache entry by the previous evaluation are reused. Nothing is approximated.
struct SrtTraceRecord {
	enum class Kind : uint8_t { Raw, Strict, Validate };
	Kind     kind;
	bool     ok;
	uint32_t word_count;
	uint32_t first_word; // index into SrtTrace::words
	uint64_t address;
	uint64_t size;
};

struct SrtTrace {
	bool                                         valid       = false;
	uint64_t                                     key_hash    = 0;
	uint64_t                                     shader_base = 0;
	std::vector<uint32_t>                        user_data;
	std::vector<SrtTraceRecord>                  records;
	std::vector<uint32_t>                        words;
	ShaderRecompiler::IR::ResourceSnapshot       resources;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	// Descriptors before validation, for rebuilding the outputs with other pass-through words.
	ShaderRecompiler::IR::RawDescriptors         raw;

	void Clear() {
		valid = false;
		records.clear();
		words.clear();
	}
	void Add(SrtTraceRecord::Kind kind, bool ok, uint64_t address, uint64_t size,
	         std::span<const uint32_t> values) {
		records.push_back({kind, ok, static_cast<uint32_t>(values.size()),
		                   static_cast<uint32_t>(words.size()), address, size});
		words.insert(words.end(), values.begin(), values.end());
	}
};

// The trace being recorded. ProgramCache runs on the GPU thread only; a thread-local rather than
// SrtRuntime::userdata because MaterializeResources swaps userdata for its own read capture.
thread_local SrtTrace* g_srt_recording = nullptr;

bool RecordStrictRead(void*, uint64_t address, std::span<uint32_t> values) {
	const bool ok = ReadShaderGuestMemory(nullptr, address, values);
	if (g_srt_recording != nullptr) {
		g_srt_recording->Add(SrtTraceRecord::Kind::Strict, ok, address, values.size_bytes(),
		                     ok ? std::span<const uint32_t>(values) : std::span<const uint32_t> {});
	}
	return ok;
}

bool RecordValidate(void*, uint64_t address, uint64_t size) {
	const bool ok = ValidateShaderGuestMemoryRange(nullptr, address, size);
	if (g_srt_recording != nullptr) {
		g_srt_recording->Add(SrtTraceRecord::Kind::Validate, ok, address, size, {});
	}
	return ok;
}

void RecordRawRead(uint64_t address, uint32_t value) {
	if (g_srt_recording != nullptr) {
		g_srt_recording->Add(SrtTraceRecord::Kind::Raw, true, address, sizeof(value), {&value, 1});
	}
}

// User data words read by the evaluation being recorded, one bit per word. ShaderParams carries at
// most 40 user data words; an index past 63 marks every word as a dependency.
thread_local uint64_t g_srt_user_data_read = 0;

void RecordUserDataRead(uint32_t index) {
	g_srt_user_data_read |= index < 64 ? uint64_t {1} << index : ~uint64_t {0};
}

// Hash of the cache inputs: the shader base, the user data count and the user data words selected
// by mask. No other word can influence the evaluation (see SourceEntry::srt_user_data_mask).
uint64_t SrtInputHash(std::span<const uint32_t> user_data, uint64_t shader_base, uint64_t mask,
                      uint32_t* words_examined) {
	std::array<uint32_t, 66> key {};
	uint32_t                 n = 0;
	key[n++]                   = static_cast<uint32_t>(user_data.size());
	key[n++]                   = static_cast<uint32_t>(shader_base >> 32u);
	for (uint64_t bits = mask; bits != 0; bits &= bits - 1) {
		const auto index = static_cast<uint32_t>(std::countr_zero(bits));
		if (index >= user_data.size()) {
			break; // past the end GetUserData fails; the count above already tells sizes apart
		}
		key[n++] = user_data[index];
	}
	*words_examined = n - 2;
	return XXH3_64bits_withSeed(key.data(), n * sizeof(uint32_t), shader_base);
}

bool SameSrtInputs(std::span<const uint32_t> a, std::span<const uint32_t> b, uint64_t mask) {
	if (a.size() != b.size()) {
		return false;
	}
	for (uint64_t bits = mask; bits != 0; bits &= bits - 1) {
		const auto index = static_cast<size_t>(std::countr_zero(bits));
		if (index >= a.size()) {
			break;
		}
		if (a[index] != b[index]) {
			return false;
		}
	}
	return true;
}

// Re-runs every recorded access in order; true when each returns exactly what it returned before.
bool SrtTraceStillValid(const SrtTrace& trace) {
	std::array<uint32_t, 64> scratch {};
	for (const auto& record: trace.records) {
		const auto expected = std::span(trace.words).subspan(record.first_word, record.word_count);
		switch (record.kind) {
			case SrtTraceRecord::Kind::Raw: {
				uint32_t word = 0;
				// Same direct read the evaluation made, including any fault-driven readback.
				std::memcpy(&word, reinterpret_cast<const void*>(record.address), sizeof(word));
				if (word != expected[0]) {
					return false;
				}
				break;
			}
			case SrtTraceRecord::Kind::Strict: {
				const auto count = record.size / sizeof(uint32_t);
				if (count > scratch.size()) {
					return false;
				}
				const bool ok =
				    ReadShaderGuestMemory(nullptr, record.address, std::span(scratch).first(count));
				if (ok != record.ok ||
				    (ok && !std::equal(expected.begin(), expected.end(), scratch.begin()))) {
					return false;
				}
				break;
			}
			case SrtTraceRecord::Kind::Validate:
				if (ValidateShaderGuestMemoryRange(nullptr, record.address, record.size) !=
				    record.ok) {
					return false;
				}
				break;
		}
	}
	return true;
}

bool SameSnapshot(const ShaderRecompiler::IR::ResourceSnapshot& a,
                  const ShaderRecompiler::IR::ResourceSnapshot& b) {
	return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
	       a.flattened_srt == b.flattened_srt && a.user_data == b.user_data &&
	       a.uniform_fill == b.uniform_fill;
}

bool SrtCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_CACHE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_SRT_CACHE_VERIFY=N re-evaluates every Nth cache hit and compares it with the cached result.
uint32_t SrtCacheVerifyInterval() {
	static const uint32_t interval = [] {
		const char* value = std::getenv("KYTY_SRT_CACHE_VERIFY");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 0;
		return static_cast<uint32_t>(std::min<unsigned long>(n, 1000000));
	}();
	return interval;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		bool                                        skip_dispatch = false;
		// Dependency traces of past evaluations, direct-mapped by a hash of the evaluation inputs:
		// one program is drawn for many objects per frame, each with its own user data.
		std::vector<SrtTrace>                       srt_traces;
		uint32_t                                    srt_current  = UINT32_MAX; // outputs held
		bool                                        srt_disabled = false; // verification mismatch
		uint32_t                                    srt_hits_since_verify = 0;
		// User data words any recorded evaluation of this program has read. Evaluation reads user
		// data only through GetUserData, so a trace's result depends on the shader base, the words
		// it read and its recorded memory accesses, never on the other words. Every trace is
		// recorded while observing its reads and the table is emptied whenever this mask grows, so
		// each stored trace's read set is inside the mask: matching the masked words (plus the
		// memory records) is as strong as matching the whole array, while draws that differ only
		// in words the resource plan never reads (per-object constants, offsets) now hit.
		uint64_t                                    srt_user_data_mask = 0;
		// Descriptor dwords that are user data words copied verbatim (inline descriptors, whose
		// address word typically changes every draw). EvaluateDescriptor does not report them as
		// reads, so they stay out of the mask; a hit substitutes the current words into the
		// trace's raw descriptors and redoes validation and specialization. Plans that cannot be
		// rematerialized key on every word instead (mask = ~0).
		struct PassThrough {
			uint8_t  kind; // 0 buffer, 1 image, 2 sampler
			uint8_t  dword;
			uint16_t index; // into info.buffers/images/samplers
			uint32_t source;
			uint32_t user_data;
		};
		bool                                        srt_prepared = false;
		bool                                        srt_rematerialize = false;
		std::vector<PassThrough>                    srt_pass_through;
		ShaderRecompiler::IR::RawDescriptors        srt_scratch;
	};

	static void PrepareSrtCache(SourceEntry& entry) {
		entry.srt_prepared      = true;
		const auto& plan        = entry.resource_plan;
		entry.srt_rematerialize = ShaderRecompiler::IR::SupportsRematerialize(plan);
		if (!entry.srt_rematerialize) {
			entry.srt_user_data_mask = ~uint64_t {0};
			return;
		}
		const auto add = [&](uint8_t kind, size_t index, uint32_t source) {
			if (source >= plan.descriptor_sources.size()) {
				return;
			}
			const auto& descriptor = plan.descriptor_sources[source];
			for (uint32_t j = 0; j < descriptor.dwords.size() && j < descriptor.dword_count; j++) {
				const auto k = ShaderRecompiler::IR::PassThroughUserData(plan, descriptor.dwords[j]);
				if (k != UINT32_MAX) {
					entry.srt_pass_through.push_back({kind, static_cast<uint8_t>(j),
					                                  static_cast<uint16_t>(index), source, k});
				}
			}
		};
		for (size_t i = 0; i < plan.info.buffers.size(); i++) {
			add(0, i, plan.info.buffers[i].source);
		}
		for (size_t i = 0; i < plan.info.images.size(); i++) {
			add(1, i, plan.info.images[i].source);
		}
		for (size_t i = 0; i < plan.info.samplers.size(); i++) {
			add(2, i, plan.info.samplers[i].source);
		}
	}

	// Writes the current pass-through words into raw; true when any of them changed.
	static bool SubstitutePassThrough(const SourceEntry&                    entry,
	                                  std::span<const uint32_t>             user_data,
	                                  ShaderRecompiler::IR::RawDescriptors& raw) {
		bool changed = false;
		for (const auto& p: entry.srt_pass_through) {
			if (!raw.active.empty() && (p.source >= raw.active.size() || raw.active[p.source] == 0)) {
				continue; // an inactive source evaluates to zero dwords, not to user data
			}
			auto& list = p.kind == 0 ? raw.buffers : p.kind == 1 ? raw.images : raw.samplers;
			auto& word = list[p.index].dwords[p.dword];
			if (word != user_data[p.user_data]) {
				word    = user_data[p.user_data];
				changed = true;
			}
		}
		return changed;
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		Common::FrameStats::CompileScope compile_scope;
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		if (const char* tmp_path = std::getenv("KYTY_TMP_SPIRV_HASHES"); tmp_path != nullptr) { // TEMP A/B hook
			uint64_t h = 1469598103934665603ull;
			for (const auto w: result.spirv) {
				h = (h ^ w) * 1099511628211ull;
			}
			static std::mutex tmp_mutex;
			std::lock_guard tmp_lock(tmp_mutex);
			if (FILE* f = std::fopen(tmp_path, "a")) {
				std::fprintf(f, "%s %016" PRIx64 " %zu %016" PRIx64 "\n", stage_name, options.shader_hash,
				             result.spirv.size(), h);
				std::fclose(f);
			}
		}

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		const auto id = ++next_shader_id;
		if (const char* traced = std::getenv("KYTY_TRACE_VS");
		    traced != nullptr &&
		    std::strstr(traced, fmt::format("{:016x}", options.shader_hash).c_str()) != nullptr) {
			Common::GpuWaitDiagnostics::TracedPrograms().push_back(id);
			std::printf("[trace-vs] %s %016" PRIx64 " is program %" PRIu64 "\n", stage_name,
			            options.shader_hash, static_cast<uint64_t>(id));
		}
		// KYTY_DUMP_SHADER=<hash>[,<hash>...]: write the disassembled SPIR-V of those shaders.
		if (const char* wanted = std::getenv("KYTY_DUMP_SHADER");
		    wanted != nullptr &&
		    std::strstr(wanted, fmt::format("{:016x}", options.shader_hash).c_str()) != nullptr) {
			spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
			std::string          text;
			if (tools.Disassemble(result.spirv, &text,
			                      SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
			                          SPV_BINARY_TO_TEXT_OPTION_INDENT)) {
				const auto name =
				    fmt::format("shader_dump_{}_{:016x}.spvasm", stage_name, options.shader_hash);
				if (FILE* f = std::fopen(name.c_str(), "w")) {
					std::fwrite(text.data(), 1, text.size(), f);
					std::fclose(f);
					std::printf("[gpu-wait] dumped %s\n", name.c_str());
				}
			}
		}
		if (Common::GpuWaitDiagnostics::Enabled()) {
			std::printf("[gpu-wait] program %llu = %s hash=0x%016llx\n",
			            static_cast<unsigned long long>(id), stage_name,
			            static_cast<unsigned long long>(options.shader_hash));
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = id, .module = module},
		};
	}

	// KYTY_SRT_STATS=1 (diagnostic): SRT cache outcomes per program, printed every 10 s, to see
	// which programs force full evaluations and which user data words vary between their misses.
	struct SrtProgramStats {
		uint64_t                     lookups = 0, hits = 0, miss_empty = 0, miss_evict = 0,
		                             miss_memory = 0;
		std::array<uint64_t, 64>     changed_words {};
		std::vector<uint32_t>        last_miss_user_data;
		std::unordered_set<uint64_t> keys;
		std::unordered_set<uint64_t> flat_contents; // flattened SRT words of evaluated misses
		std::unordered_set<uint64_t> outputs;       // full outputs of evaluated misses
		uint64_t                     mask = 0;
		size_t                       user_data_size = 0;
	};

	static std::unordered_map<uint64_t, SrtProgramStats>& SrtStatsMap() {
		static std::unordered_map<uint64_t, SrtProgramStats> stats;
		return stats;
	}

	// After an evaluated miss: how many different table contents and outputs a program really has.
	static void RecordSrtOutputs(uint64_t program_hash, const SourceEntry& entry) {
		static const bool enabled = std::getenv("KYTY_SRT_STATS") != nullptr;
		if (!enabled) {
			return;
		}
		auto&       s   = SrtStatsMap()[program_hash];
		const auto& res = entry.resources;
		const auto  flat =
		    XXH3_64bits(res.flattened_srt.data(), res.flattened_srt.size() * sizeof(uint32_t));
		if (s.flat_contents.size() < 65536) {
			s.flat_contents.insert(flat);
		}
		if (s.outputs.size() < 65536) {
			uint64_t   h   = flat;
			const auto mix = [&](const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
				for (const auto& d: values) {
					h = XXH3_64bits_withSeed(d.dwords.data(), d.dword_count * sizeof(uint32_t), h);
				}
			};
			mix(res.buffers);
			mix(res.images);
			mix(res.samplers);
			s.outputs.insert(h);
		}
	}

	static void RecordSrtStats(uint64_t program_hash, const SourceEntry& entry,
	                           const SrtTrace& trace, uint64_t key_hash,
	                           std::span<const uint32_t> user_data, int outcome) {
		static const bool enabled = std::getenv("KYTY_SRT_STATS") != nullptr;
		if (!enabled) {
			return;
		}
		auto&       stats      = SrtStatsMap();
		static auto last_print = std::chrono::steady_clock::now();
		auto&       s          = stats[program_hash];
		s.lookups++;
		s.mask           = entry.srt_user_data_mask;
		s.user_data_size = user_data.size();
		if (s.keys.size() < 65536) {
			s.keys.insert(key_hash);
		}
		if (outcome == 0) {
			s.hits++;
		} else {
			if (outcome == 1) {
				(trace.valid ? s.miss_evict : s.miss_empty)++;
			} else {
				s.miss_memory++;
			}
			for (size_t i = 0; i < user_data.size() && i < 64; i++) {
				if (i < s.last_miss_user_data.size() && s.last_miss_user_data[i] != user_data[i]) {
					s.changed_words[i]++;
				}
			}
			s.last_miss_user_data.assign(user_data.begin(), user_data.end());
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - last_print < std::chrono::seconds(10)) {
			return;
		}
		last_print = now;
		std::vector<std::pair<uint64_t, const SrtProgramStats*>> order;
		uint64_t total_lookups = 0, total_misses = 0;
		for (const auto& [hash, value]: stats) {
			order.emplace_back(hash, &value);
			total_lookups += value.lookups;
			total_misses += value.lookups - value.hits;
		}
		std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
			return a.second->lookups - a.second->hits > b.second->lookups - b.second->hits;
		});
		std::printf("[srt-stats] %zu programs, %" PRIu64 " lookups, %" PRIu64 " misses\n",
		            stats.size(), total_lookups, total_misses);
		for (size_t i = 0; i < order.size() && i < 12; i++) {
			const auto& v = *order[i].second;
			std::string changed;
			std::array<uint32_t, 64> idx {};
			for (uint32_t w = 0; w < 64; w++) {
				idx[w] = w;
			}
			std::sort(idx.begin(), idx.end(),
			          [&](uint32_t a, uint32_t b) { return v.changed_words[a] > v.changed_words[b]; });
			for (int k = 0; k < 5 && v.changed_words[idx[k]] != 0; k++) {
				changed += fmt::format(" w{}:{}", idx[k], v.changed_words[idx[k]]);
			}
			std::printf("[srt-stats]   %016" PRIx64 " lookups %" PRIu64 " hit %.0f%% empty %" PRIu64
			            " evict %" PRIu64 " memory %" PRIu64 " keys %zu flat %zu outputs %zu"
			            " mask %d/%zu changed:%s\n",
			            order[i].first, v.lookups, 100.0 * static_cast<double>(v.hits) / v.lookups,
			            v.miss_empty, v.miss_evict, v.miss_memory, v.keys.size(),
			            v.flat_contents.size(), v.outputs.size(),
			            std::popcount(v.mask), v.user_data_size, changed.c_str());
		}
		std::fflush(stdout);
	}

	// KYTY_SRT_DUMP=<program hash>[,...] (diagnostic): the expressions a program's descriptors, SRT
	// reads and resource control flow evaluate, printed once.
	static void DumpSrtPlan(uint64_t program_hash, const SourceEntry& entry) {
		static const std::string wanted = [] {
			const char* value = std::getenv("KYTY_SRT_DUMP");
			return std::string(value != nullptr ? value : "");
		}();
		if (wanted.empty() || wanted.find(fmt::format("{:016x}", program_hash)) == std::string::npos) {
			return;
		}
		static std::unordered_set<uint64_t> dumped;
		if (!dumped.insert(program_hash).second) {
			return;
		}
		using ShaderRecompiler::IR::DescribeSrtValue;
		const auto& plan = entry.resource_plan;
		std::printf("[srt-dump] %016" PRIx64 " user_data_base=%u sources=%zu srt_reads=%zu "
		            "control_flow=%zu buffers=%zu images=%zu samplers=%zu\n",
		            program_hash, plan.user_data_base, plan.descriptor_sources.size(),
		            plan.srt_reads.size(), plan.control_flow.size(), plan.info.buffers.size(),
		            plan.info.images.size(), plan.info.samplers.size());
		const auto source = [&](const char* kind, size_t index, uint32_t s) {
			if (s >= plan.descriptor_sources.size()) {
				return;
			}
			const auto& d = plan.descriptor_sources[s];
			for (uint32_t j = 0; j < d.dword_count && j < d.dwords.size(); j++) {
				std::printf("[srt-dump]   %s %zu (source %u) dw%u = %s\n", kind, index, s, j,
				            DescribeSrtValue(plan, d.dwords[j]).c_str());
			}
		};
		for (size_t i = 0; i < plan.info.buffers.size(); i++) {
			source("buffer", i, plan.info.buffers[i].source);
		}
		for (size_t i = 0; i < plan.info.images.size(); i++) {
			source("image", i, plan.info.images[i].source);
		}
		for (size_t i = 0; i < plan.info.samplers.size(); i++) {
			source("sampler", i, plan.info.samplers[i].source);
		}
		for (size_t i = 0; i < plan.srt_reads.size(); i++) {
			std::printf("[srt-dump]   srt_read %zu (flat %u) = %s\n", i, plan.srt_reads[i].flat_offset,
			            DescribeSrtValue(plan, plan.srt_reads[i].value).c_str());
		}
		for (size_t i = 0; i < plan.control_flow.size(); i++) {
			if (!plan.control_flow[i].condition.IsEmpty()) {
				std::printf("[srt-dump]   block %zu condition = %s\n", i,
				            DescribeSrtValue(plan, plan.control_flow[i].condition).c_str());
			}
		}
		std::fflush(stdout);
	}

	// MaterializeResources for a draw, skipped when the entry's dependency trace shows the previous
	// evaluation (whose outputs are still in the entry) would repeat exactly.
	static void Materialize(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                        uint64_t program_hash) {
		DumpSrtPlan(program_hash, entry);
		constexpr uint32_t Slots   = 256;
		const bool         enabled = SrtCacheEnabled() && !entry.srt_disabled;
		SrtTrace*          trace   = nullptr;
		uint32_t           slot    = 0;
		if (enabled) {
			if (entry.srt_traces.empty()) {
				entry.srt_traces.resize(Slots);
			}
			if (!entry.srt_prepared) {
				PrepareSrtCache(entry);
			}
			uint32_t       examined = 0;
			const uint64_t key_hash = SrtInputHash(runtime.user_data, runtime.shader_base,
			                                       entry.srt_user_data_mask, &examined);
			Common::FrameStats::g_srt_dependency_words.fetch_add(examined,
			                                                     std::memory_order_relaxed);
			slot  = static_cast<uint32_t>(key_hash & (Slots - 1u));
			trace = &entry.srt_traces[slot];
			const bool same_inputs =
			    trace->valid && trace->key_hash == key_hash &&
			    trace->shader_base == runtime.shader_base &&
			    SameSrtInputs(trace->user_data, runtime.user_data, entry.srt_user_data_mask);
			if (!same_inputs) {
				Common::FrameStats::g_srt_miss_inputs.fetch_add(1, std::memory_order_relaxed);
				RecordSrtStats(program_hash, entry, *trace, key_hash, runtime.user_data, 1);
			} else if (!SrtTraceStillValid(*trace)) {
				Common::FrameStats::g_srt_miss_memory.fetch_add(1, std::memory_order_relaxed);
				RecordSrtStats(program_hash, entry, *trace, key_hash, runtime.user_data, 2);
			} else {
				Common::FrameStats::g_srt_hits.fetch_add(1, std::memory_order_relaxed);
				RecordSrtStats(program_hash, entry, *trace, key_hash, runtime.user_data, 0);
				Common::FrameStats::g_srt_checked_reads.fetch_add(trace->records.size(),
				                                                  std::memory_order_relaxed);
				bool substituted = false;
				if (entry.srt_rematerialize && !entry.srt_pass_through.empty()) {
					// The pass-through words are all in range: the trace evaluated them.
					entry.srt_scratch = trace->raw;
					substituted = SubstitutePassThrough(entry, runtime.user_data, entry.srt_scratch);
				}
				if (substituted) {
					Common::FrameStats::g_srt_substituted.fetch_add(1, std::memory_order_relaxed);
					entry.resources      = trace->resources; // flattened SRT and uniform fill
					entry.specialization = trace->specialization;
					EXIT_IF(!ShaderRecompiler::IR::RematerializeResources(
					    entry.resource_plan, runtime, entry.srt_scratch, entry.resources,
					    entry.specialization));
					entry.srt_current = UINT32_MAX; // no longer the trace's own outputs
				} else {
					if (entry.srt_current != slot) {
						entry.resources      = trace->resources;
						entry.specialization = trace->specialization;
						entry.srt_current    = slot;
					}
					// The snapshot also carries a plain copy of the user data (push data, draw
					// offsets): the one output that follows every word, so refresh it.
					if (!std::ranges::equal(entry.resources.user_data, runtime.user_data)) {
						entry.resources.user_data.assign(runtime.user_data.begin(),
						                                 runtime.user_data.end());
					}
				}
				const auto verify = SrtCacheVerifyInterval();
				if (verify == 0 || ++entry.srt_hits_since_verify < verify) {
					return;
				}
				entry.srt_hits_since_verify = 0;
				ShaderRecompiler::IR::ResourceSnapshot       fresh;
				ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
				EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
				                                                    fresh, fresh_specialization));
				if (SameSnapshot(fresh, entry.resources) &&
				    fresh_specialization == entry.specialization) {
					return;
				}
				Common::FrameStats::g_srt_mismatches.fetch_add(1, std::memory_order_relaxed);
				std::printf("SRT CACHE MISMATCH: program=0x%016" PRIx64 " base=0x%016" PRIx64
				            " user_data=%zu recorded_reads=%zu mask=0x%016" PRIx64
				            "; caching disabled for this program\n",
				            program_hash, runtime.shader_base, runtime.user_data.size(),
				            trace->records.size(), entry.srt_user_data_mask);
				entry.srt_disabled = true;
				entry.srt_traces.clear();
				entry.srt_current    = UINT32_MAX;
				entry.resources      = std::move(fresh);
				entry.specialization = std::move(fresh_specialization);
				return;
			}
			trace->Clear();
			trace->key_hash    = key_hash;
			trace->shader_base = runtime.shader_base;
			trace->user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
		}

		static const bool tmp_fast_verify = std::getenv("KYTY_TMP_SRT_FAST_VERIFY") != nullptr; // TEMP
		ShaderRecompiler::IR::ResourceSnapshot       tmp_res;
		ShaderRecompiler::IR::ResourceSpecialization tmp_spec;
		bool                                         tmp_ok = false;
		if (tmp_fast_verify) {
			ShaderRecompiler::IR::SetSrtFastPaths(false);
			tmp_ok = ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime, tmp_res,
			                                                    tmp_spec);
			ShaderRecompiler::IR::SetSrtFastPaths(true);
		}
		Common::FrameStats::g_srt_evaluations.fetch_add(1, std::memory_order_relaxed);
		auto recording = runtime;
		if (trace != nullptr) {
			recording.read_specialization_memory = RecordStrictRead;
			recording.validate_memory_range      = RecordValidate;
			recording.observe_raw_read           = RecordRawRead;
			recording.observe_user_data          = RecordUserDataRead;
			g_srt_recording                      = trace;
			g_srt_user_data_read                 = 0;
		}
		const bool ok = ShaderRecompiler::IR::MaterializeResources(
		    entry.resource_plan, recording, entry.resources, entry.specialization,
		    trace != nullptr && entry.srt_rematerialize ? &trace->raw : nullptr);
		g_srt_recording = nullptr;
		if (tmp_fast_verify) { // TEMP
			static uint64_t checks = 0, mismatches = 0;
			checks++;
			if (tmp_ok != ok || !SameSnapshot(tmp_res, entry.resources) ||
			    !(tmp_spec == entry.specialization)) {
				mismatches++;
				if (mismatches <= 20) {
					std::printf("[tmp-srt-verify] MISMATCH program=%016" PRIx64 " ok=%d/%d\n",
					            program_hash, int(tmp_ok), int(ok));
				}
			}
			if ((checks & 0x3fff) == 0) {
				std::printf("[tmp-srt-verify] checks=%" PRIu64 " mismatches=%" PRIu64 "\n", checks,
				            mismatches);
			}
		}
		EXIT_IF(!ok);
		RecordSrtOutputs(program_hash, entry);
		entry.srt_current = UINT32_MAX;
		if (trace != nullptr) {
			if ((g_srt_user_data_read & ~entry.srt_user_data_mask) != 0) {
				// This evaluation read a word no stored trace was keyed on. Widen the mask, drop
				// every other trace (their keys and slots were computed without that word) and
				// file this one under the widened key.
				entry.srt_user_data_mask |= g_srt_user_data_read;
				Common::FrameStats::g_srt_mask_resets.fetch_add(1, std::memory_order_relaxed);
				uint32_t       unused  = 0;
				const uint64_t widened = SrtInputHash(runtime.user_data, runtime.shader_base,
				                                      entry.srt_user_data_mask, &unused);
				const auto     widened_slot = static_cast<uint32_t>(widened & (Slots - 1u));
				for (uint32_t i = 0; i < Slots; i++) {
					if (i != slot) {
						entry.srt_traces[i].Clear();
					}
				}
				if (widened_slot != slot) {
					std::swap(entry.srt_traces[widened_slot], entry.srt_traces[slot]);
				}
				slot            = widened_slot;
				trace           = &entry.srt_traces[slot];
				trace->key_hash = widened;
			}
			trace->resources      = entry.resources;
			trace->specialization = entry.specialization;
			trace->valid          = true;
			entry.srt_current     = slot;
		}
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		if (entry != programs.end() && entry->second.skip_dispatch) {
			return {};
		}
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
			.validate_memory_range      = ValidateShaderGuestMemoryRange,
		};
		if (entry != programs.end()) {
			Materialize(entry->second, runtime, params.hash);
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations,
			        [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == entry->second.specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = params.user_data;
		options.back_code   = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		auto translated = [&] {
			Common::FrameStats::TimeScope translate_scope(Common::FrameStats::g_translate_us);
			Common::FrameStats::g_translate_count.fetch_add(1, std::memory_order_relaxed);
			static const int repeat = [] { // TEMP profiling hook
				const char* v = std::getenv("KYTY_TMP_TRANSLATE_REPEAT");
				return v != nullptr ? std::atoi(v) : 0;
			}();
			for (int i = 0; i < repeat; i++) {
				(void)ShaderRecompiler::TranslateProgram(params.code, options);
			}
			return ShaderRecompiler::TranslateProgram(params.code, options);
		}();
		if (translated.skip_dispatch) {
			entry = programs.try_emplace(lookup_key, ShaderRecompiler::IR::ResourcePlan {}).first;
			entry->second.skip_dispatch = true;
			return {};
		}
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			Materialize(entry->second, runtime, params.hash);
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = PathUtil::GetPath(PathUtil::PIPELINE_CACHE_DIR) / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr) {
		return;
	}
	WriteSnapshotLocked();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::MaybeSnapshotLocked() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_snapshot_pending = true;
	constexpr auto Interval = std::chrono::seconds(30);
	const auto     now      = std::chrono::steady_clock::now();
	if (now - m_last_snapshot < Interval) {
		return;
	}
	m_last_snapshot = now;
	WriteSnapshotLocked();
}

void PipelineCache::WriteSnapshotLocked() {
	if (m_driver_cache == nullptr || !m_snapshot_pending) {
		return;
	}
	m_snapshot_pending = false;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)", vk::to_string(result),
		                 size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; }) &&
		           ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0]) ==
		               BlendMappingSupport::SourceAlpha) {
			// Preserve logical alpha when the export mapping moves it.
			pixel_info.alpha_blend_source_remap = true;
			pixel_info.dual_source_blending     = true;
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = {};
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                           = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                           = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		const bool alpha_remap =
		    slot == 0 && ps_input_info != nullptr && ps_input_info->alpha_blend_source_remap;
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && !alpha_remap &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (alpha_remap) {
			static_params.blend_alpha_source_remap = true;
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list             = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back          = !rect_list && mc.cull_back;
	static_params.cull_front         = !rect_list && mc.cull_front;
	static_params.face               = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	MaybeSnapshotLocked();

	return *iter->second;
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                                           const ShaderProgram& compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	MaybeSnapshotLocked();
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
