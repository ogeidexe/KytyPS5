#include "graphics/host_gpu/renderer/gpuPassProfiler.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::GpuPassProfiler {

namespace {

constexpr uint32_t PairCount = 16384;

struct Pending {
	uint32_t pair = 0;
	uint64_t tick = 0;
	uint64_t key  = 0;
};

struct Totals {
	std::string label;
	uint64_t    count = 0;
	double      ms    = 0.0;
};

struct State {
	vk::Device                              device = nullptr;
	vk::QueryPool                           pool   = nullptr;
	double                                  ns_per_tick = 1.0;
	uint32_t                                next        = 0;
	std::vector<Pending>                    pending;
	std::unordered_map<uint64_t, Totals>    totals;
	std::chrono::steady_clock::time_point   last_report = std::chrono::steady_clock::now();
	std::unordered_map<std::string, uint64_t> restarts;
	std::unordered_map<std::string, uint64_t> transitions;
};

State& GetState() {
	static State state;
	return state;
}

void Collect(CommandScheduler& scheduler) {
	auto& state  = GetState();
	auto& master = scheduler.GetMasterSemaphore();
	master.Refresh();
	size_t kept = 0;
	for (const auto& entry: state.pending) {
		if (!master.IsFree(entry.tick)) {
			state.pending[kept++] = entry;
			continue;
		}
		uint64_t   stamps[2] {};
		const auto result = state.device.getQueryPoolResults(
		    state.pool, entry.pair * 2, 2, sizeof(stamps), stamps, sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eSuccess && stamps[1] >= stamps[0]) {
			auto& total = state.totals[entry.key];
			total.count++;
			total.ms += static_cast<double>(stamps[1] - stamps[0]) * state.ns_per_tick / 1e6;
		}
	}
	state.pending.resize(kept);

	const auto now = std::chrono::steady_clock::now();
	if (now - state.last_report < std::chrono::seconds(10)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - state.last_report).count();
	state.last_report    = now;
	std::vector<const Totals*> order;
	double                     sum = 0.0;
	for (const auto& [key, total]: state.totals) {
		order.push_back(&total);
		sum += total.ms;
	}
	std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->ms > b->ms; });
	std::printf("[gpu-pass] %.1f s: %.1f GPU ms/s measured in %zu kinds\n", seconds, sum / seconds,
	            order.size());
	for (size_t i = 0; i < order.size() && i < 30; i++) {
		const auto& t = *order[i];
		std::printf("[gpu-pass]   %6.2f ms/s %5.1f%% %8" PRIu64 " x %7.1f us  %s\n", t.ms / seconds,
		            sum > 0 ? 100.0 * t.ms / sum : 0.0, t.count,
		            t.count != 0 ? 1000.0 * t.ms / static_cast<double>(t.count) : 0.0,
		            t.label.c_str());
	}
	std::vector<std::pair<std::string, uint64_t>> restart_order(state.restarts.begin(),
	                                                            state.restarts.end());
	std::sort(restart_order.begin(), restart_order.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < restart_order.size() && i < 12; i++) {
		std::printf("[gpu-pass]   restart %8.1f/s after end at %s\n",
		            static_cast<double>(restart_order[i].second) / seconds,
		            restart_order[i].first.c_str());
	}
	std::vector<std::pair<std::string, uint64_t>> transition_order(state.transitions.begin(),
	                                                               state.transitions.end());
	std::sort(transition_order.begin(), transition_order.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < transition_order.size() && i < 12; i++) {
		std::printf("[gpu-pass]   transition %8.1f/s %s\n",
		            static_cast<double>(transition_order[i].second) / seconds,
		            transition_order[i].first.c_str());
	}
	std::fflush(stdout);
	state.totals.clear();
	state.restarts.clear();
	state.transitions.clear();
}

} // namespace

void NoteRestart(const std::source_location& where) {
	std::string file = where.file_name();
	if (const auto pos = file.find("renderer"); pos != std::string::npos) {
		file = file.substr(pos + 9);
	}
	GetState().restarts[file + ":" + std::to_string(where.line()) + " " + where.function_name()]++;
}

void NoteTransition(const std::string& what) {
	GetState().transitions[what]++;
}

bool Enabled() {
	static const bool enabled = std::getenv("KYTY_GPU_PASS_PROFILE") != nullptr;
	return enabled;
}

uint32_t Begin(const CommandBuffer& buffer, uint64_t key, const std::string& label) {
	if (!Enabled()) {
		return UINT32_MAX;
	}
	auto& state     = GetState();
	auto& scheduler = buffer.GetContext().GetCommandScheduler();
	if (state.pool == nullptr) {
		auto& graphics = buffer.GetGraphics();
		state.device   = graphics.device;
		state.ns_per_tick =
		    static_cast<double>(graphics.physical_device_properties.limits.timestampPeriod);
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = PairCount * 2;
		if (state.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
			state.pool = nullptr;
			return UINT32_MAX;
		}
	}
	Collect(scheduler);
	if (state.pending.size() >= PairCount - 1) {
		return UINT32_MAX;
	}
	const auto pair = state.next;
	state.next      = (state.next + 1) % PairCount;
	auto& total     = state.totals[key];
	if (total.label.empty()) {
		total.label = label;
	}
	const auto command = buffer.Handle();
	command.resetQueryPool(state.pool, pair * 2, 2);
	command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, state.pool, pair * 2);
	state.pending.push_back({pair, UINT64_MAX, key});
	return pair;
}

void End(const CommandBuffer& buffer, uint32_t token) {
	if (token == UINT32_MAX) {
		return;
	}
	auto& state = GetState();
	buffer.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                               token * 2 + 1);
	const auto tick = buffer.GetContext().GetCommandScheduler().CurrentTick();
	for (auto it = state.pending.rbegin(); it != state.pending.rend(); ++it) {
		if (it->pair == token && it->tick == UINT64_MAX) {
			it->tick = tick;
			break;
		}
	}
}

} // namespace Libs::Graphics::GpuPassProfiler
