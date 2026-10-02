#ifndef KYTY_COMMON_SYSTEM_INFO_H_
#define KYTY_COMMON_SYSTEM_INFO_H_

#include <cstdint>
#include <string>

namespace Common {

struct SystemInfo {
	std::string ProcessorName;
};

[[nodiscard]] SystemInfo GetSystemInfo();

// Host memory of this process (diagnostics). Zero where the platform does not report a field.
struct ProcessMemoryInfo {
	uint64_t private_bytes  = 0; // committed private memory (excludes the guest memory sections)
	uint64_t working_set    = 0;
	uint64_t heap_committed = 0; // the process heap, which serves malloc/new
	uint64_t heap_allocated = 0;
};

[[nodiscard]] ProcessMemoryInfo GetProcessMemoryInfo();

} // namespace Common

#endif /* KYTY_COMMON_SYSTEM_INFO_H_ */
