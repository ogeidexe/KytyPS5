#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <source_location>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	// asynchronous: commands are recorded into streams that a recording thread replays into
	// Vulkan command buffers and submits, in order, so the caller's thread does not pay for the
	// driver's recording and submission. The renderer's scheduler is asynchronous unless
	// KYTY_ASYNC_RECORD=0.
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool asynchronous = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering(std::source_location where = std::source_location::current());
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Non-blocking form: true when every priority operation deferred at or before tick has run.
	[[nodiscard]] bool        PriorityOperationsDone(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	[[nodiscard]] bool             Asynchronous() const noexcept { return m_asynchronous; }
	// Returns once the recording thread has submitted every batch queued so far. Anything else
	// that submits to the queue must call this first (see SubmitOrdering), so its work cannot
	// precede, or wait on, work still queued here.
	void WaitRecordingIdle();
	// Called by every queue submission that does not go through an asynchronous scheduler.
	static void SubmitOrdering(const CommandScheduler* submitter);

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// Queued when its stream is opened: the recording thread replays the stream while the GPU
	// thread is still recording it, and submits once Submit() has closed it.
	struct RecordedBatch {
		std::unique_ptr<CommandStream> stream;
		bool                           closed = false; // the fields below are set
		SubmitInfo                     submit;
		uint64_t                       tick         = 0;
		uint32_t                       debug_op     = 0;
		uint64_t                       debug_submit = 0;
		uint32_t                       debug_args[4] {};
		uint64_t                       debug_arg4   = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	void RecordingThread(std::stop_token stop);
	bool RecordBatch(RecordedBatch& batch, const std::stop_token& stop);
	void PublishCommands();
	void QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick, uint32_t debug_op,
	                 uint64_t debug_submit, const uint32_t* debug_args, uint64_t debug_arg4);
	std::unique_ptr<CommandStream> AcquireStream();
	void                           StopRecordingThread();

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;

	// Asynchronous recording (GPU thread produces batches, the recording thread consumes them).
	bool                                        m_asynchronous = false;
	// Elements stay in place until the recording thread has submitted them (deque references
	// survive push_back and pop_front), so the GPU thread can close its open batch in place.
	std::deque<RecordedBatch>                   m_batches;
	RecordedBatch*                              m_open_batch = nullptr; // GPU thread
	std::vector<std::unique_ptr<CommandStream>> m_free_streams;
	std::mutex                                  m_batch_mutex;
	std::condition_variable                     m_batch_available;
	std::condition_variable                     m_batch_submitted;
	uint64_t                                    m_last_queued_tick = 0;
	uint64_t                                    m_submitted_tick   = 0;
	std::jthread                                m_recording_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
