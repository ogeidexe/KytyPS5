#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
using SrtMemoryRangeValidator = bool (*)(void* userdata, uint64_t address, uint64_t size);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	SrtMemoryRangeValidator   validate_memory_range      = nullptr;
	// Told about every word read directly from guest memory (the read_memory == nullptr path).
	// It observes only and cannot change the value, so evaluation is unaffected.
	void (*observe_raw_read)(uint64_t address, uint32_t value) = nullptr;
	// Told the index of every user data word the evaluation reads (GetUserData is the only way it
	// reads user data). Observes only, like observe_raw_read.
	void (*observe_user_data)(uint32_t index) = nullptr;
	std::span<const uint32_t> workgroup_counts;
};

enum class RuntimeValueType { Any, Integer };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Whether SRT evaluation takes its direct paths for flattened-SRT reads and descriptor dwords
// (identical results; KYTY_SRT_FAST_PATHS=0 starts with them off). GPU thread only.
void SetSrtFastPaths(bool enabled);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);
// The user data index a descriptor dword copies verbatim (the value is GetUserData itself), or
// UINT32_MAX. EvaluateDescriptor reads such dwords without reporting them to observe_user_data.
uint32_t PassThroughUserData(const ResourcePlan& program, Value value);

// Diagnostics: the expression a plan value evaluates, user data words shown as ud[N] (relative to
// the plan's user data base), depth-limited.
std::string DescribeSrtValue(const ResourcePlan& program, Value value, int max_depth = 10);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateDirectRead(const ResourcePlan::DirectSrtRead& read, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
	// The flat SRT buffer this session's RefreshFlatBuffer filled, once it succeeded.
	const std::vector<uint32_t>*     m_flat = nullptr;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
