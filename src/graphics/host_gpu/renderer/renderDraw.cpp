#include "graphics/host_gpu/renderer/renderDraw.h"
#include "common/gpuWaitDiagnostics.h"
#include "graphics/host_gpu/gpuCheckpoints.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	vk_buffer.setViewportWithCount(viewport_count, viewports.data());
	vk_buffer.setScissorWithCount(viewport_count, scissors.data());

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	vk_buffer.setLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	vk_buffer.setBlendConstants(blend_constants.data());
	vk_buffer.setDepthTestEnable(depth.depth_test_enable ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthWriteEnable(depth.depth_write_enable ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthCompareOp(depth.depth_compare_op);

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	vk_buffer.setStencilTestEnable(depth.stencil_test_enable ? VK_TRUE : VK_FALSE);
	if (depth.stencil_test_enable) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state) {
			vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			vk_buffer.setStencilCompareMask(face, state.compareMask);
			vk_buffer.setStencilWriteMask(face, state.writeMask);
			vk_buffer.setStencilReference(face, state.reference);
		};
		set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
		set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	if (rendering.num_color_attachments != 0) {
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable);
	}
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (sh_regs.target_output_mode[slot] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

// KYTY_TRACE_VS only: GPU-side copies of every range a traced draw reads, taken by the command
// buffer right before the draw, so after a device loss the inputs of the draw the GPU could not
// finish can be compared with earlier draws of the same mesh. Host-visible memory stays readable
// after the loss. The copies are small (one mesh) and cost no CPU scanning.
struct TracedDrawCapture {
	static constexpr uint32_t Slots     = 48;
	static constexpr uint64_t SlotBytes = 1024 * 1024;
	struct Range {
		char     kind  = 0; // 'v' vertex buffer, 'i' index buffer, 's' storage buffer
		uint32_t index = 0;
		uint64_t guest = 0, size = 0, copied = 0, slot_offset = 0, host_offset = 0;
		VkBuffer host  = VK_NULL_HANDLE;
	};
	struct Attribute {
		uint32_t fields[4] {};
		int      attr_id = -1, buffer_index = 0, registers_num = 0;
		uint32_t fetch_index = 0;
	};
	struct Slot {
		uint64_t               draw = 0, submit = 0, tick = 0;
		uint32_t               marker = 0, index_count = 0, instances = 0, first_instance = 0;
		uint64_t               vs = 0, ps = 0, index_address = 0;
		int32_t                vertex_offset = 0;
		uint32_t               index_type = 0;
		std::vector<Range>     ranges;
		std::vector<Attribute> attributes;
		std::vector<uint32_t>  strides, num_records;
		std::string            state;
		std::vector<std::string> images;
	};
	std::unique_ptr<Buffer>    storage;
	std::array<Slot, Slots>    slots;
	uint64_t                   next = 0;
};

// KYTY_TRACE_STATS=1 (with the default full trace mode): fragment shader invocations and samples
// passed (fragments that survived kill and the depth test, i.e. late depth writes) per traced
// draw, to compare the draw's real fragment workload with the standalone replay's.
struct TracedDrawStats {
	static constexpr uint32_t Count = 256;
	vk::QueryPool             statistics;
	vk::QueryPool             occlusion;
	uint64_t                  issued = 0;
	uint64_t                  read   = 0;
	double                    sum[3] {};
	double                    max[3] {};
	// Which traced draw each query measured (trace draw number, submit, checkpoint marker).
	uint64_t                  draw[Count] {};
	uint64_t                  submit[Count] {};
	uint32_t                  marker[Count] {};
};

static TracedDrawStats* GetTracedDrawStats(vk::Device device) {
	static const bool enabled = std::getenv("KYTY_TRACE_STATS") != nullptr;
	if (!enabled) {
		return nullptr;
	}
	static TracedDrawStats stats = [device] {
		TracedDrawStats s;
		vk::QueryPoolCreateInfo info {};
		info.queryType          = vk::QueryType::ePipelineStatistics;
		info.queryCount         = TracedDrawStats::Count;
		info.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
		                          vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
		s.statistics            = device.createQueryPool(info).value;
		info.queryType          = vk::QueryType::eOcclusion;
		info.pipelineStatistics = {};
		s.occlusion             = device.createQueryPool(info).value;
		return s;
	}();
	return &stats;
}

// Collects the result of the query about to be reused (written TracedDrawStats::Count traced draws
// ago, so long finished unless the GPU is stuck) and prints a running summary.
static void CollectTracedDrawStats(vk::Device device, TracedDrawStats& s, uint32_t index) {
	if (s.issued < TracedDrawStats::Count) {
		return;
	}
	uint64_t st[3] {}, occ[2] {};
	const auto flags = vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability;
	if (device.getQueryPoolResults(s.statistics, index, 1, sizeof(st), st, sizeof(st), flags) !=
	        vk::Result::eSuccess ||
	    device.getQueryPoolResults(s.occlusion, index, 1, sizeof(occ), occ, sizeof(occ), flags) !=
	        vk::Result::eSuccess ||
	    st[2] == 0 || occ[1] == 0) {
		return;
	}
	const double v[3] = {static_cast<double>(st[0]), static_cast<double>(st[1]),
	                     static_cast<double>(occ[0])};
	for (int i = 0; i < 3; i++) {
		s.sum[i] += v[i];
		s.max[i] = std::max(s.max[i], v[i]);
	}
	// KYTY_TRACE_STATS_OUTLIER=<fragment shader invocations> (default 2,000,000, ~50x the normal
	// skinning draw): each traced draw above it is printed on its own, so a stall that completes
	// shows whether it was this draw doing vastly more fragment work than usual.
	static const double outlier = [] {
		const char* value = std::getenv("KYTY_TRACE_STATS_OUTLIER");
		return value != nullptr ? std::strtod(value, nullptr) : 2'000'000.0;
	}();
	if (v[1] > outlier) {
		std::printf("[trace-stats] OUTLIER trace draw %" PRIu64 " (submit %" PRIu64
		            ", marker %u): primitives %.0f, fragment invocations %.0f, samples passed %.0f\n",
		            s.draw[index], s.submit[index], s.marker[index], v[0], v[1], v[2]);
		std::fflush(stdout);
	}
	if (++s.read % 500 == 0) {
		std::printf("[trace-stats] %" PRIu64 " traced draws: mean primitives %.0f, fragment "
		            "invocations %.0f, samples passed %.0f; max %.0f / %.0f / %.0f\n",
		            s.read, s.sum[0] / s.read, s.sum[1] / s.read, s.sum[2] / s.read, s.max[0],
		            s.max[1], s.max[2]);
		std::fflush(stdout);
	}
}

static TracedDrawCapture& GetTracedDrawCapture() {
	static TracedDrawCapture capture;
	return capture;
}

static void DumpTracedDrawCapture() {
	auto& capture = GetTracedDrawCapture();
	if (!capture.storage) {
		return;
	}
	const auto* base = capture.storage->Mapped().data();
	const auto  dir  = std::string("trace_capture");
	(void)std::system(("mkdir " + dir + " 2>nul").c_str());
	FILE* summary = std::fopen((dir + "/summary.txt").c_str(), "w");
	if (summary == nullptr) {
		return;
	}
	for (uint32_t i = 0; i < TracedDrawCapture::Slots; i++) {
		const auto& slot = capture.slots[i];
		if (slot.draw == 0) {
			continue;
		}
		std::fprintf(summary,
		             "draw %" PRIu64 " slot %u marker %u submit %" PRIu64 " tick %" PRIu64
		             " vs %" PRIu64 " ps %" PRIu64 " indices %u instances %u first_instance %u"
		             " vertex_offset %d index_type %u index_address 0x%" PRIx64 "\n",
		             slot.draw, i, slot.marker, slot.submit, slot.tick, slot.vs, slot.ps,
		             slot.index_count, slot.instances, slot.first_instance, slot.vertex_offset,
		             slot.index_type, slot.index_address);
		std::fprintf(summary, "  state %s\n", slot.state.c_str());
		for (const auto& image: slot.images) {
			std::fprintf(summary, "  image %s\n", image.c_str());
		}
		for (size_t b = 0; b < slot.strides.size(); b++) {
			std::fprintf(summary, "  vbuf %zu stride %u num_records %u\n", b, slot.strides[b],
			             slot.num_records[b]);
		}
		for (const auto& a: slot.attributes) {
			std::fprintf(summary,
			             "  attr %d buffer %d regs %d fetch %u vsharp %08x %08x %08x %08x\n",
			             a.attr_id, a.buffer_index, a.registers_num, a.fetch_index, a.fields[0],
			             a.fields[1], a.fields[2], a.fields[3]);
		}
		for (const auto& r: slot.ranges) {
			const auto name = dir + "/d" + std::to_string(slot.draw) + "_" + r.kind +
			                  std::to_string(r.index) + ".bin";
			std::fprintf(summary,
			             "  range %c%u guest 0x%" PRIx64 " size 0x%" PRIx64 " copied 0x%" PRIx64
			             " host %p+0x%" PRIx64 " file %s\n",
			             r.kind, r.index, r.guest, r.size, r.copied, static_cast<void*>(r.host),
			             r.host_offset, name.c_str());
			if (FILE* f = std::fopen(name.c_str(), "wb")) {
				std::fwrite(base + uint64_t {i} * TracedDrawCapture::SlotBytes + r.slot_offset, 1,
				            r.copied, f);
				std::fclose(f);
			}
		}
	}
	std::fclose(summary);
	std::printf("[trace-vs] wrote %s/summary.txt\n", dir.c_str());
	std::fflush(stdout);
}

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandRecorder&       vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandRecorder& vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			if (Common::GpuWaitDiagnostics::Enabled()) {
				char header[160];
				std::snprintf(header, sizeof(header),
				              "mesh limit: groups=%u instances=%u index_count=%u indexed=%d "
				              "first_instance=%u",
				              mesh_groups, draw.instance_count, draw.index_count,
				              draw.IsIndexed() ? 1 : 0, draw.first_instance);
				Common::GpuWaitDiagnostics::Dump(header);
			}
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	Common::GpuWaitDiagnostics::CurrentOp() = {'D', state.programs.vertex[0].id,
	                                           state.programs.pixel.id, submit_id,
	                                           GpuCheckpoints::GetState().next.load() & 0x3ffffu};
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs);
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	// KYTY_TRACE_MODE (diagnostic, traced draws only) separates what the capture below changes in
	// the command stream: "split" only ends the rendering scope before the draw, "barrier" also
	// records the two full barriers, "none" changes nothing (CPU-side bookkeeping only), default
	// "full" also copies every range the draw reads.
	int32_t                  stats_query = -1; // KYTY_TRACE_STATS query of this draw
	static const std::string trace_mode = [] {
		const char* value = std::getenv("KYTY_TRACE_MODE");
		return std::string(value != nullptr ? value : "full");
	}();
	if (trace_mode != "full" &&
	    Common::GpuWaitDiagnostics::IsTracedProgram(state.programs.vertex[0].id)) {
		if (trace_mode == "split" || trace_mode == "barrier") {
			m_context.GetCommandScheduler().EndRendering();
		}
		if (trace_mode == "barrier") {
			vk::MemoryBarrier before {};
			before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
			before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0,
			                          nullptr, 0, nullptr);
			vk::MemoryBarrier after {};
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eHostRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                          vk::PipelineStageFlagBits::eAllCommands |
			                              vk::PipelineStageFlagBits::eHost,
			                          {}, 1, &after, 0, nullptr, 0, nullptr);
		}
	} else if (Common::GpuWaitDiagnostics::IsTracedProgram(state.programs.vertex[0].id)) {
		auto& capture = GetTracedDrawCapture();
		if (!capture.storage) {
			capture.storage = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::Download, 0,
			    vk::BufferUsageFlagBits::eTransferDst,
			    TracedDrawCapture::Slots * TracedDrawCapture::SlotBytes);
			Common::GpuWaitDiagnostics::OnDeviceLost() = [] { DumpTracedDrawCapture(); };
		}
		static uint64_t traced_draws = 0;
		const auto      slot_index   = capture.next++ % TracedDrawCapture::Slots;
		auto&           slot         = capture.slots[slot_index];
		slot                         = {};
		slot.draw                    = ++traced_draws;
		slot.submit                  = submit_id;
		slot.tick                    = m_context.GetCommandScheduler().CurrentTick();
		slot.marker                  = GpuCheckpoints::GetState().next.load();
		slot.vs                      = state.programs.vertex[0].id;
		slot.ps                      = state.programs.pixel.id;
		slot.index_count             = draw.index_count;
		slot.instances               = draw.instance_count;
		slot.first_instance          = draw.first_instance;
		slot.vertex_offset           = static_cast<int32_t>(emit.vertex_offset);
		slot.index_type              = static_cast<uint32_t>(index_binding.type);
		slot.index_address           = index_source.address;
		{
			const auto& regs = buffer.GetRegisters();
			const auto& clip = regs.GetClipControl();
			const auto& vp   = regs.GetScreenViewport().viewports[0];
			const auto& dc   = regs.GetDepthControl();
			char        text[1024];
			std::snprintf(text, sizeof(text),
			              "dx_clip_space=%d zclip_near_disable=%d zclip_far_disable=%d "
			              "clip_disable=%d vtx_kill_or=%d clip_err_detect_disable=%d ucp=%u "
			              "host_depth_clip=%d viewport scale=(%g,%g,%g) offset=(%g,%g,%g) "
			              "z=[%g,%g] z_enable=%d z_write=%d zfunc=%u",
			              clip.dx_clip_space ? 1 : 0, clip.min_z_clip_disable ? 1 : 0,
			              clip.max_z_clip_disable ? 1 : 0, clip.clip_disable ? 1 : 0,
			              clip.vertex_kill_any ? 1 : 0, clip.cull_on_clipping_error_disable ? 1 : 0,
			              static_cast<unsigned>(clip.user_clip_planes), clip.IsZClipEnabled() ? 1 : 0,
			              vp.xscale, vp.yscale, vp.zscale, vp.xoffset, vp.yoffset, vp.zoffset, vp.zmin,
			              vp.zmax, dc.z_enable ? 1 : 0, dc.z_write_enable ? 1 : 0,
			              static_cast<unsigned>(dc.zfunc));
			slot.state = text;
			const auto& ds = rendering.depth_stencil_attachment;
			std::snprintf(text, sizeof(text),
			              " attachment_view=%p attachment_layout=%d attachment_clear=%d"
			              " render_layers=%u render=%ux%u marker=%u",
			              static_cast<void*>(static_cast<VkImageView>(ds.image_view)),
			              static_cast<int>(ds.image_layout), ds.depth_clear ? 1 : 0,
			              rendering.num_layers, rendering.width, rendering.height,
			              GpuCheckpoints::GetState().next.load() & 0x3ffffu);
			slot.state += text;
			const auto& mode = regs.GetModeControl();
			const auto& po   = regs.GetPolyOffset();
			std::snprintf(text, sizeof(text),
			              " cull_front=%d cull_back=%d face=%d poly_mode=%u bias_front=%d bias_back=%d"
			              " bias_float_fmt=%d neg_db_bits=%d bias_clamp=%g front_scale=%g"
			              " front_offset=%g back_scale=%g back_offset=%g stencil=%d depth_bounds=%d"
			              " write_enable=%d compare=%d",
			              mode.cull_front ? 1 : 0, mode.cull_back ? 1 : 0, mode.face ? 1 : 0,
			              static_cast<unsigned>(mode.poly_mode), mode.poly_offset_front_enable ? 1 : 0,
			              mode.poly_offset_back_enable ? 1 : 0, po.db_is_float_fmt ? 1 : 0,
			              static_cast<int>(po.neg_num_db_bits), po.clamp, po.front_scale,
			              po.front_offset, po.back_scale, po.back_offset,
			              state.depth_info.stencil_test_enable ? 1 : 0,
			              state.depth_info.depth_bounds_test_enable ? 1 : 0,
			              state.depth_info.depth_write_enable ? 1 : 0,
			              static_cast<int>(state.depth_info.depth_compare_op));
			slot.state += text;
			{
				// Raw guest registers with a GFX10 decode of DB_SHADER_CONTROL.
				const auto& sh  = regs.GetShaderRegisters();
				const auto  dsc = sh.db_shader_control.raw;
				const auto& z   = regs.GetDepthRenderTarget();
				std::snprintf(
				    text, sizeof(text),
				    " DB_SHADER_CONTROL=0x%08x{z_export=%u z_order=%u kill=%u cov_to_mask=%u"
				    " mask_export=%u exec_on_hier_fail=%u exec_on_noop=%u depth_before_shader=%u"
				    " conservative_z=%u dual_quad_disable=%u pops=%u}"
				    " DB_RENDER_CONTROL=0x%08x DB_RENDER_OVERRIDE=0x%08x SPI_SHADER_Z_FORMAT=0x%08x"
				    " SPI_PS_INPUT_ENA=0x%08x SPI_PS_INPUT_ADDR=0x%08x z_htile=%d z_expclear=%d"
				    " z_slice=%u..%u htile_base=0x%" PRIx64 " z_read=0x%" PRIx64 " z_write=0x%" PRIx64,
				    dsc, dsc & 1u, (dsc >> 4) & 3u, (dsc >> 6) & 1u, (dsc >> 7) & 1u, (dsc >> 8) & 1u,
				    (dsc >> 9) & 1u, (dsc >> 10) & 1u, (dsc >> 12) & 1u, (dsc >> 13) & 3u,
				    (dsc >> 15) & 1u, (dsc >> 16) & 1u, regs.GetRenderControl().raw,
				    regs.GetDepthRenderOverride().raw, sh.shader_z_format, sh.ps_input_ena,
				    sh.ps_input_addr, z.z_info.htile_acceleration ? 1 : 0,
				    z.z_info.expclear_enabled ? 1 : 0, static_cast<unsigned>(z.depth_view.slice_start),
				    static_cast<unsigned>(z.depth_view.slice_max),
				    static_cast<uint64_t>(z.htile_data_base_addr),
				    static_cast<uint64_t>(z.z_read_base_addr), static_cast<uint64_t>(z.z_write_base_addr));
				slot.state += text;
			}
			{
				// Log each distinct bias configuration once, so the values are known even from
				// runs that do not lose the device.
				static std::vector<std::string> seen;
				char key[256];
				std::snprintf(key, sizeof(key),
				              "render=%ux%u bias_front=%d bias_back=%d clamp=%g fs=%g fo=%g bs=%g bo=%g"
				              " float=%d bits=%d cull=%d/%d",
				              rendering.width, rendering.height, mode.poly_offset_front_enable ? 1 : 0,
				              mode.poly_offset_back_enable ? 1 : 0, po.clamp, po.front_scale,
				              po.front_offset, po.back_scale, po.back_offset, po.db_is_float_fmt ? 1 : 0,
				              static_cast<int>(po.neg_num_db_bits), mode.cull_front ? 1 : 0,
				              mode.cull_back ? 1 : 0);
				if (seen.size() < 64 && std::find(seen.begin(), seen.end(), key) == seen.end()) {
					seen.emplace_back(key);
					std::printf("[trace-vs] depth state %s\n", key);
				}
			}
		}
		{
			auto&      textures = m_context.GetTextureCache();
			const auto describe = [&](const char* use, ImageId id, vk::ImageView view,
			                          vk::ImageLayout layout, const ImageViewInfo* view_info) {
				char text[512];
				const auto* image = id ? textures.m_slot_images.try_get(id) : nullptr;
				if (image == nullptr) {
					std::snprintf(text, sizeof(text), "%s id=%u:%u missing", use, id.index,
					              id.generation);
				} else {
					const auto& info  = image->info;
					const auto  found = textures.m_image_ticks.find(id);
					const auto  ticks = found != textures.m_image_ticks.end()
					                        ? found->second
					                        : TextureCache::ImageTicks {};
					std::snprintf(
					    text, sizeof(text),
					    "%s id=%u:%u registered=%d guest=0x%" PRIx64 "+0x%" PRIx64
					    " extent=%ux%ux%u levels=%u layers=%u vkfmt=%d guestfmt=%d tile=%d"
					    " image=%p view=%p layout=%d view_fmt=%d view_levels=%u+%u view_layers=%u+%u"
					    " created_tick=%" PRIu64 " init_tick=%" PRIu64 " now_tick=%" PRIu64,
					    use, id.index, id.generation, image->registered ? 1 : 0, info.data.address,
					    info.data.size, info.extent.width, info.extent.height, info.extent.depth,
					    info.resources.levels, info.resources.layers,
					    static_cast<int>(info.pixel_format), static_cast<int>(info.guest_format),
					    static_cast<int>(info.tile_mode),
					    static_cast<void*>(static_cast<VkImage>(image->backing.image)),
					    static_cast<void*>(static_cast<VkImageView>(view)), static_cast<int>(layout),
					    view_info ? static_cast<int>(view_info->format) : -1,
					    view_info ? view_info->base_level : 0, view_info ? view_info->level_count : 0,
					    view_info ? view_info->base_layer : 0, view_info ? view_info->layer_count : 0,
					    ticks.created, ticks.initialized, m_context.GetCommandScheduler().CurrentTick());
				}
				slot.images.emplace_back(text);
			};
			for (const auto* stage: stages) {
				for (const auto& texture: stage->images) {
					describe("texture", texture.image_id, texture.image_view, texture.layout,
					         &texture.desc.view_info);
				}
			}
			describe("depth", state.depth_info.image_id, nullptr, vk::ImageLayout::eUndefined,
			         &state.depth_info.desc.view_info);
			for (uint32_t i = 0; i < state.color_count; i++) {
				describe("color", state.color_info[i].image_id, nullptr,
				         vk::ImageLayout::eUndefined, nullptr);
			}
		}
		const auto& vs = state.vertex_info[0];
		for (int i = 0; i < vs.buffers_num; i++) {
			slot.strides.push_back(vs.buffers[i].stride);
			slot.num_records.push_back(vs.buffers[i].num_records);
		}
		for (int i = 0; i < vs.resources_num; i++) {
			TracedDrawCapture::Attribute a;
			std::memcpy(a.fields, vs.resources[i].fields, sizeof(a.fields));
			a.attr_id       = vs.resources_dst[i].attr_id;
			a.buffer_index  = vs.resources_dst[i].buffer_index;
			a.registers_num = vs.resources_dst[i].registers_num;
			a.fetch_index   = vs.resources_dst[i].fetch_index;
			slot.attributes.push_back(a);
		}
		m_context.GetCommandScheduler().EndRendering();
		if (auto* stats = GetTracedDrawStats(m_context.GetGraphics().device); stats != nullptr) {
			stats_query = static_cast<int32_t>(stats->issued++ % TracedDrawStats::Count);
			CollectTracedDrawStats(m_context.GetGraphics().device, *stats,
			                       static_cast<uint32_t>(stats_query));
			vk_buffer.resetQueryPool(stats->statistics, stats_query, 1);
			vk_buffer.resetQueryPool(stats->occlusion, stats_query, 1);
			stats->draw[stats_query]   = slot.draw;
			stats->submit[stats_query] = submit_id;
			stats->marker[stats_query] = slot.marker;
		}
		const auto dst    = capture.storage->Handle();
		uint64_t   cursor = 0;
		std::vector<std::pair<vk::Buffer, vk::BufferCopy>> copies;
		const auto add = [&](char kind, uint32_t index, uint64_t guest, vk::Buffer host,
		                     uint64_t offset, uint64_t size) {
			if (!host || size == 0 || size == VK_WHOLE_SIZE) {
				return;
			}
			TracedDrawCapture::Range r;
			r.kind        = kind;
			r.index       = index;
			r.guest       = guest;
			r.size        = size;
			r.copied      = std::min(size, TracedDrawCapture::SlotBytes - cursor);
			r.slot_offset = cursor;
			r.host        = static_cast<VkBuffer>(host);
			r.host_offset = offset;
			if (r.copied == 0) {
				return;
			}
			copies.push_back({host, vk::BufferCopy {offset,
			                                        slot_index * TracedDrawCapture::SlotBytes + cursor,
			                                        r.copied}});
			cursor = Common::AlignUp<uint64_t>(cursor + r.copied, 16);
			slot.ranges.push_back(r);
		};
		for (const auto& info: stages.front()->buffers) {
			const auto j = static_cast<uint32_t>(&info - stages.front()->buffers.data());
			const auto guest = j < stages.front()->buffer_sources.size()
			                       ? stages.front()->buffer_sources[j].address
			                       : 0;
			add('s', j, guest, info.buffer, info.offset, info.range);
		}
		// Pixel stage storage buffers ('p'), for replaying the draw outside the emulator.
		if (stages.size() > 1) {
			const auto* pixel = stages.back();
			for (const auto& info: pixel->buffers) {
				const auto j     = static_cast<uint32_t>(&info - pixel->buffers.data());
				const auto guest = j < pixel->buffer_sources.size() ? pixel->buffer_sources[j].address
				                                                    : 0;
				add('p', j, guest, info.buffer, info.offset, info.range);
			}
		}
		if (draw.IsIndexed()) {
			const uint64_t element = index_binding.type == vk::IndexType::eUint32 ? 4u : 2u;
			add('i', 0, index_source.address, index_binding.buffer, index_binding.offset,
			    uint64_t {draw.index_count} * element);
		}
		for (uint32_t i = 0; i < vertex_bindings.count; i++) {
			add('v', i, i < static_cast<uint32_t>(vs.buffers_num) ? vs.buffers[i].addr : 0,
			    vertex_bindings.buffers[i], vertex_bindings.offsets[i], vertex_bindings.sizes[i]);
		}
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		for (const auto& [source, region]: copies) {
			vk_buffer.copyBuffer(source, dst, 1, &region);
		}
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eHostRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eAllCommands |
		                              vk::PipelineStageFlagBits::eHost,
		                          {}, 1, &after, 0, nullptr, 0, nullptr);
	}
	{
		// KYTY_GPU_HAZARD_VERIFY only (a no-op otherwise).
		auto& cache = m_context.GetBufferCache();
		for (uint32_t i = 0; i < vertex_bindings.count; i++) {
			cache.VerifyBindingAlive(vertex_bindings.buffers[i], "vertex buffer", submit_id);
		}
		cache.VerifyBindingAlive(index_binding.buffer, "index buffer", submit_id);
		auto& textures = m_context.GetTextureCache();
		for (const auto* stage: stages) {
			for (const auto& info: stage->buffers) {
				cache.VerifyBindingAlive(info.buffer, "storage buffer", submit_id);
			}
			for (const auto& texture: stage->images) {
				textures.VerifyImageAlive(texture.image_id, "texture", submit_id);
			}
		}
		for (uint32_t i = 0; i < state.color_count; i++) {
			textures.VerifyImageAlive(state.color_info[i].image_id, "color target", submit_id);
		}
		textures.VerifyImageAlive(state.depth_info.image_id, "depth target", submit_id);
	}
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info, rendering);
	if (Common::GpuWaitDiagnostics::IsTracedProgram(state.programs.vertex[0].id)) {
		// KYTY_DIAG_DEPTH=nowrite|nobias|always: diagnostic-only overrides of the traced draws'
		// dynamic depth state, to test which part of the depth path the device loss depends on.
		static const std::string experiment = [] {
			const char* value = std::getenv("KYTY_DIAG_DEPTH");
			return std::string(value != nullptr ? value : "");
		}();
		if (experiment == "nowrite") {
			vk_buffer.setDepthWriteEnable(VK_FALSE);
		} else if (experiment == "nobias") {
			vk_buffer.setDepthBiasEnable(VK_FALSE);
		} else if (experiment == "always") {
			vk_buffer.setDepthCompareOp(vk::CompareOp::eAlways);
		}
	}
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	// Diagnostics only (no-op otherwise): the draw about to be emitted, with its programs.
	GpuCheckpoints::Mark(vk_buffer, {0xD0u, submit_id, static_cast<uint32_t>(state.programs.vertex[0].id),
	                                 static_cast<uint32_t>(state.programs.pixel.id),
	                                 mesh_active ? mesh_groups : 0u, draw.instance_count,
	                                 draw.index_count});
	static const bool diag_skip = [] {
		const char* value = std::getenv("KYTY_DIAG_DEPTH");
		return value != nullptr && std::strcmp(value, "skip") == 0;
	}();
	auto* stats = stats_query >= 0 ? GetTracedDrawStats(m_context.GetGraphics().device) : nullptr;
	if (stats != nullptr) {
		vk_buffer.beginQuery(stats->statistics, stats_query, {});
		vk_buffer.beginQuery(stats->occlusion, stats_query, vk::QueryControlFlagBits::ePrecise);
	}
	if (diag_skip && Common::GpuWaitDiagnostics::IsTracedProgram(state.programs.vertex[0].id)) {
		// KYTY_DIAG_DEPTH=skip: diagnostic only, the traced draws are not emitted at all.
	} else if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	if (stats != nullptr) {
		vk_buffer.endQuery(stats->occlusion, stats_query);
		vk_buffer.endQuery(stats->statistics, stats_query);
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
