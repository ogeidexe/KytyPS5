#include "heapProfiler.h"

#include "common/common.h"

#include <cstddef>
#include <cstdlib>
#include <new>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <malloc.h>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

// The replacement operator new/delete below serve every C++ allocation in the executable. With
// profiling off they are exactly malloc/free (and _aligned_malloc/_aligned_free), like the CRT's
// own. With it on, roughly one allocation per sampling interval of allocated bytes records its
// call stack; a side table of the sampled pointers lets their frees subtract them again. Nothing
// is stored next to the allocations, so blocks may still cross the CRT DLL boundary either way.

namespace {

constexpr int      MaxFrames       = 32;
constexpr uint64_t DefaultInterval = 256 * 1024;

std::atomic<int> g_state {-1}; // -1: not read yet, 0: off, 1: on

bool Enabled() noexcept {
	int state = g_state.load(std::memory_order_relaxed);
	if (state < 0) {
		state = std::getenv("KYTY_HEAP_PROFILE") != nullptr ? 1 : 0;
		g_state.store(state, std::memory_order_relaxed);
	}
	return state == 1;
}

uint64_t Interval() noexcept {
	static const uint64_t interval = [] {
		const char* value = std::getenv("KYTY_HEAP_PROFILE_INTERVAL");
		const auto  n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
		return n != 0 ? n : DefaultInterval;
	}();
	return interval;
}

struct Stack {
	void*    frames[MaxFrames] {};
	uint32_t depth = 0;
	uint32_t hash  = 0;
};

struct Record {
	Stack    stack;
	int64_t  live_bytes    = 0; // estimated: each live sample stands for max(size, interval)
	int64_t  live_samples  = 0;
	uint64_t total_samples = 0;
};

struct Sample {
	uint32_t record = 0;
	uint64_t weight = 0;
};

// Guarded by g_lock, created on the first sample. Allocations made while t_busy is set (the
// profiler's own) are neither sampled nor looked up when freed.
SRWLOCK                                             g_lock    = SRWLOCK_INIT;
std::vector<Record>*                                g_records = nullptr;
std::unordered_map<uint32_t, std::vector<uint32_t>>* g_by_hash = nullptr;
std::unordered_map<void*, Sample>*                  g_samples = nullptr;

thread_local bool     t_busy      = false;
thread_local int64_t  t_countdown = 0;
thread_local uint64_t t_random    = 0;

// One bit per pointer hash, set when a pointer is sampled: most frees see a clear bit and skip
// the lock. Bits are never cleared, which only costs an occasional needless lookup.
constexpr uint32_t                 BloomBits = 24;
std::atomic<uint64_t>              g_bloom[(size_t {1} << BloomBits) / 64];

uint64_t BloomIndex(void* p) noexcept {
	return ((reinterpret_cast<uintptr_t>(p) >> 4u) * 0x9E3779B97F4A7C15ull) >> (64u - BloomBits);
}

int64_t NextStep() noexcept {
	if (t_random == 0) {
		t_random = reinterpret_cast<uintptr_t>(&t_random) ^ GetTickCount64() ^ 0x2545F4914F6CDD1Dull;
	}
	t_random ^= t_random << 13u;
	t_random ^= t_random >> 7u;
	t_random ^= t_random << 17u;
	const auto interval = Interval();
	return static_cast<int64_t>(interval / 2 + t_random % interval); // mean: one interval
}

void OnAlloc(void* p, size_t size) noexcept {
	if (p == nullptr || t_busy || !Enabled()) {
		return;
	}
	t_countdown -= static_cast<int64_t>(size);
	if (t_countdown > 0) {
		return;
	}
	t_countdown = NextStep();
	t_busy      = true;
	Stack stack;
	ULONG hash  = 0;
	stack.depth = RtlCaptureStackBackTrace(2, MaxFrames, stack.frames, &hash);
	stack.hash  = hash;
	const auto weight = std::max<uint64_t>(size, Interval());
	AcquireSRWLockExclusive(&g_lock);
	if (g_records == nullptr) {
		g_records = new std::vector<Record>();
		g_by_hash = new std::unordered_map<uint32_t, std::vector<uint32_t>>();
		g_samples = new std::unordered_map<void*, Sample>();
	}
	uint32_t index = UINT32_MAX;
	auto&    same  = (*g_by_hash)[stack.hash];
	for (const auto candidate: same) {
		const auto& other = (*g_records)[candidate].stack;
		if (other.depth == stack.depth &&
		    std::equal(other.frames, other.frames + other.depth, stack.frames)) {
			index = candidate;
			break;
		}
	}
	if (index == UINT32_MAX) {
		index = static_cast<uint32_t>(g_records->size());
		g_records->push_back({stack});
		same.push_back(index);
	}
	auto& record = (*g_records)[index];
	record.live_bytes += static_cast<int64_t>(weight);
	record.live_samples++;
	record.total_samples++;
	(*g_samples)[p] = {index, weight};
	const auto bit  = BloomIndex(p);
	g_bloom[bit / 64].fetch_or(uint64_t {1} << (bit % 64), std::memory_order_relaxed);
	ReleaseSRWLockExclusive(&g_lock);
	t_busy = false;
}

// Called before the block is released, so a concurrent allocation reusing the address cannot be
// recorded before this one is removed.
void OnFree(void* p) noexcept {
	if (p == nullptr || t_busy || g_state.load(std::memory_order_relaxed) != 1) {
		return;
	}
	const auto bit = BloomIndex(p);
	if ((g_bloom[bit / 64].load(std::memory_order_relaxed) & (uint64_t {1} << (bit % 64))) == 0) {
		return;
	}
	t_busy = true;
	AcquireSRWLockExclusive(&g_lock);
	if (g_samples != nullptr) {
		if (const auto found = g_samples->find(p); found != g_samples->end()) {
			auto& record = (*g_records)[found->second.record];
			record.live_bytes -= static_cast<int64_t>(found->second.weight);
			record.live_samples--;
			g_samples->erase(found);
		}
	}
	ReleaseSRWLockExclusive(&g_lock);
	t_busy = false;
}

void* Allocate(size_t size) {
	if (size == 0) {
		size = 1;
	}
	for (;;) {
		if (void* p = std::malloc(size); p != nullptr) {
			OnAlloc(p, size);
			return p;
		}
		const auto handler = std::get_new_handler();
		if (handler == nullptr) {
			throw std::bad_alloc();
		}
		handler();
	}
}

void* AllocateAligned(size_t size, std::align_val_t alignment) {
	if (size == 0) {
		size = 1;
	}
	for (;;) {
		if (void* p = _aligned_malloc(size, static_cast<size_t>(alignment)); p != nullptr) {
			OnAlloc(p, size);
			return p;
		}
		const auto handler = std::get_new_handler();
		if (handler == nullptr) {
			throw std::bad_alloc();
		}
		handler();
	}
}

void Free(void* p) noexcept {
	OnFree(p);
	std::free(p);
}

void FreeAligned(void* p) noexcept {
	OnFree(p);
	_aligned_free(p);
}

std::string Describe(HANDLE process, DWORD64 pc, DWORD inline_context) {
	char  buffer[sizeof(SYMBOL_INFO) + 512] {};
	auto* symbol         = reinterpret_cast<SYMBOL_INFO*>(buffer);
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
	symbol->MaxNameLen   = 511;
	DWORD64    displacement = 0;
	const bool named        = inline_context != 0
	                              ? SymFromInlineContext(process, pc, inline_context, &displacement, symbol)
	                              : SymFromAddr(process, pc, &displacement, symbol);
	std::string text = named ? symbol->Name : "?";
	IMAGEHLP_LINE64 line {sizeof(line)};
	DWORD           line_displacement = 0;
	const bool      has_line =
        inline_context != 0
	        ? SymGetLineFromInlineContext(process, pc, inline_context, 0, &line_displacement, &line)
	        : SymGetLineFromAddr64(process, pc, &line_displacement, &line);
	if (has_line && line.FileName != nullptr) {
		std::string file = line.FileName;
		if (const auto pos = file.find("src\\"); pos != std::string::npos) {
			file = file.substr(pos + 4);
		} else if (const auto slash = file.find_last_of("\\/"); slash != std::string::npos) {
			file = file.substr(slash + 1);
		}
		text += "  (" + file + ":" + std::to_string(line.LineNumber) + ")";
	}
	return text;
}

} // namespace

namespace Kyty {

void HeapProfileWriteReport() {
	const char* path = std::getenv("KYTY_HEAP_PROFILE");
	if (path == nullptr || g_state.load(std::memory_order_relaxed) != 1) {
		return;
	}
	t_busy = true;
	std::vector<Record> records;
	size_t              live_pointers = 0;
	AcquireSRWLockExclusive(&g_lock);
	if (g_records != nullptr) {
		records       = *g_records;
		live_pointers = g_samples->size();
	}
	ReleaseSRWLockExclusive(&g_lock);
	std::sort(records.begin(), records.end(),
	          [](const Record& a, const Record& b) { return a.live_bytes > b.live_bytes; });
	int64_t total = 0;
	for (const auto& record: records) {
		total += record.live_bytes;
	}

	static HANDLE process = [] {
		HANDLE handle = GetCurrentProcess();
		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
		SymInitialize(handle, nullptr, TRUE);
		return handle;
	}();
	static int report = 0;
	const auto name   = std::string(path) + "." + std::to_string(report++) + ".txt";
	if (FILE* f = std::fopen(name.c_str(), "w"); f != nullptr) {
		std::fprintf(f, "estimated live bytes %.1f MB in %zu stacks, %zu sampled pointers live, interval %llu\n",
		             static_cast<double>(total) / (1024.0 * 1024.0), records.size(), live_pointers,
		             static_cast<unsigned long long>(Interval()));
		for (size_t i = 0; i < records.size() && i < 80; i++) {
			const auto& record = records[i];
			if (record.live_bytes <= 0) {
				break;
			}
			std::fprintf(f, "\n#%zu live %.2f MB (%lld samples live, %llu total) stack %08x\n", i,
			             static_cast<double>(record.live_bytes) / (1024.0 * 1024.0),
			             static_cast<long long>(record.live_samples),
			             static_cast<unsigned long long>(record.total_samples), record.stack.hash);
			for (uint32_t frame = 0; frame < record.stack.depth; frame++) {
				// A return address: look up the call instruction before it.
				const auto pc    = reinterpret_cast<DWORD64>(record.stack.frames[frame]) - 1;
				const auto count = SymAddrIncludeInlineTrace(process, pc);
				DWORD      context = 0;
				DWORD      index   = 0;
				if (count != 0 && SymQueryInlineTrace(process, pc, 0, pc, pc, &context, &index)) {
					for (DWORD k = 0; k < count; k++) {
						std::fprintf(f, "    %s\n", Describe(process, pc, context + k).c_str());
					}
				}
				std::fprintf(f, "    %s\n", Describe(process, pc, 0).c_str());
			}
		}
		std::fclose(f);
	}
	t_busy = false;
}

} // namespace Kyty

void* operator new(size_t size) {
	return Allocate(size);
}
void* operator new[](size_t size) {
	return Allocate(size);
}
void* operator new(size_t size, const std::nothrow_t&) noexcept {
	try {
		return Allocate(size);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t size, const std::nothrow_t&) noexcept {
	try {
		return Allocate(size);
	} catch (...) {
		return nullptr;
	}
}
void* operator new(size_t size, std::align_val_t alignment) {
	return AllocateAligned(size, alignment);
}
void* operator new[](size_t size, std::align_val_t alignment) {
	return AllocateAligned(size, alignment);
}
void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
	try {
		return AllocateAligned(size, alignment);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
	try {
		return AllocateAligned(size, alignment);
	} catch (...) {
		return nullptr;
	}
}
void operator delete(void* p) noexcept {
	Free(p);
}
void operator delete[](void* p) noexcept {
	Free(p);
}
void operator delete(void* p, size_t) noexcept {
	Free(p);
}
void operator delete[](void* p, size_t) noexcept {
	Free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
	Free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
	Free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
	FreeAligned(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
	FreeAligned(p);
}
void operator delete(void* p, size_t, std::align_val_t) noexcept {
	FreeAligned(p);
}
void operator delete[](void* p, size_t, std::align_val_t) noexcept {
	FreeAligned(p);
}
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
	FreeAligned(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
	FreeAligned(p);
}

#else

namespace Kyty {

void HeapProfileWriteReport() {}

} // namespace Kyty

#endif
