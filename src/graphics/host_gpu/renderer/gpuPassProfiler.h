#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GPU_PASS_PROFILER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GPU_PASS_PROFILER_H_

#include <cstdint>
#include <source_location>
#include <string>

namespace Libs::Graphics {

class CommandBuffer;

// KYTY_GPU_PASS_PROFILE=1 (diagnostic): GPU time of every host render pass and compute dispatch,
// measured with timestamp queries and summed per kind of work (render target size and attachment
// count, or compute shader). Every 10 s the GPU thread prints the kinds that took the most GPU
// time. GPU thread only. Passes overlap on the GPU, so the sum can exceed wall time; it ranks work.
namespace GpuPassProfiler {

[[nodiscard]] bool Enabled();
// Brackets the work recorded into buffer until End. Must be called outside dynamic rendering.
// Returns a token for End, or UINT32_MAX when nothing is measured.
[[nodiscard]] uint32_t Begin(const CommandBuffer& buffer, uint64_t key, const std::string& label);
void                   End(const CommandBuffer& buffer, uint32_t token);
// A render pass began on exactly the targets of the pass ended by where.
void NoteRestart(const std::source_location& where);
// An image layout transition that had to end the open render pass.
void NoteTransition(const std::string& what);

} // namespace GpuPassProfiler

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_GPU_PASS_PROFILER_H_ */
