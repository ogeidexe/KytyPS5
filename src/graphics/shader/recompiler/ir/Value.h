#pragma once

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/Reg.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Block;
class Inst;

class Value {
public:
	Value() = default;
	explicit Value(Inst* value);
	explicit Value(ScalarReg value);
	explicit Value(VectorReg value);
	explicit Value(bool value);
	explicit Value(uint8_t value);
	explicit Value(uint16_t value);
	explicit Value(uint32_t value);
	explicit Value(uint64_t value);

	static Value F16(uint16_t bits);
	static Value F32(float value);

	[[nodiscard]] bool IsEmpty() const;
	[[nodiscard]] bool IsImmediate() const;
	[[nodiscard]] bool IsIdentity() const;
	[[nodiscard]] bool IsPhi() const;
	[[nodiscard]] Type GetType() const;

	[[nodiscard]] Inst*     Instruction() const;
	[[nodiscard]] Inst*     TryInstruction() const;
	[[nodiscard]] Inst*     ResolveInstruction() const;
	[[nodiscard]] Value     Resolve() const;
	[[nodiscard]] ScalarReg ScalarRegister() const;
	[[nodiscard]] VectorReg VectorRegister() const;
	[[nodiscard]] bool      U1() const;
	[[nodiscard]] uint8_t   U8() const;
	[[nodiscard]] uint16_t  U16() const;
	[[nodiscard]] uint32_t  U32() const;
	[[nodiscard]] uint64_t  U64() const;
	[[nodiscard]] uint16_t  F16Bits() const;
	[[nodiscard]] float     F32Value() const;

	bool operator==(const Value& other) const;

private:
	Type type = Type::Void;
	union {
		Inst*     inst;
		ScalarReg scalar_reg;
		VectorReg vector_reg;
		bool      imm_u1;
		uint8_t   imm_u8;
		uint16_t  imm_u16;
		uint32_t  imm_u32;
		uint64_t  imm_u64;
	};

	explicit Value(Type type, uint64_t bits);
};
static_assert(std::is_trivially_copyable_v<Value>);

template <Type type_>
class TypedValue: public Value {
public:
	TypedValue() = default;

	template <Type other_type>
	requires(TypesOverlap(type_, other_type))
	TypedValue(const TypedValue<other_type>& value): Value(value) {}

	explicit TypedValue(Value value): Value(value) {
		EXIT_IF(!AreTypesCompatible(value.GetType(), type_));
	}
};

using U1     = TypedValue<Type::U1>;
using U8     = TypedValue<Type::U8>;
using U16    = TypedValue<Type::U16>;
using U32    = TypedValue<Type::U32>;
using U64    = TypedValue<Type::U64>;
using F16    = TypedValue<Type::F16>;
using F32    = TypedValue<Type::F32>;
using U32F32 = TypedValue<Type::U32 | Type::F32>;

struct Use {
	Inst*  user    = nullptr;
	size_t operand = 0;

	bool operator==(const Use&) const = default;
};

// An instruction's users, in the order they were added. Most values have one or two, which live
// inside the instruction: translation adds a use for every operand it writes, and a heap
// allocation each time was a large part of the translation cost.
class UseList {
public:
	UseList() noexcept: m_inline {} {}
	UseList(const UseList& other): UseList() {
		Reserve(other.m_size);
		std::copy_n(other.data(), other.m_size, data());
		m_size = other.m_size;
	}
	UseList(UseList&& other) noexcept: UseList() {
		if (other.m_capacity != 0) {
			m_heap           = other.m_heap;
			m_capacity       = other.m_capacity;
			other.m_inline   = {};
			other.m_capacity = 0;
		} else {
			m_inline = other.m_inline;
		}
		m_size       = other.m_size;
		other.m_size = 0;
	}
	UseList& operator=(const UseList&) = delete;
	UseList& operator=(UseList&&)      = delete;
	~UseList() {
		if (m_capacity != 0) {
			delete[] m_heap;
		}
	}

	[[nodiscard]] size_t     size() const noexcept { return m_size; }
	[[nodiscard]] bool       empty() const noexcept { return m_size == 0; }
	[[nodiscard]] Use*       data() noexcept { return m_capacity != 0 ? m_heap : m_inline.data(); }
	[[nodiscard]] const Use* data() const noexcept {
		return m_capacity != 0 ? m_heap : m_inline.data();
	}
	[[nodiscard]] Use*       begin() noexcept { return data(); }
	[[nodiscard]] Use*       end() noexcept { return data() + m_size; }
	[[nodiscard]] const Use* begin() const noexcept { return data(); }
	[[nodiscard]] const Use* end() const noexcept { return data() + m_size; }

	void push_back(const Use& use) {
		Reserve(m_size + size_t {1});
		data()[m_size++] = use;
	}
	void erase(const Use* position) noexcept {
		Use* const first = data() + (position - data());
		std::copy(first + 1, end(), first);
		m_size--;
	}
	void clear() noexcept { m_size = 0; }

private:
	static constexpr size_t InlineCapacity = 2;

	void Reserve(size_t count) {
		const size_t capacity = m_capacity != 0 ? m_capacity : InlineCapacity;
		if (count <= capacity) {
			return;
		}
		const size_t grown = std::max(count, capacity * 2);
		auto*        heap  = new Use[grown];
		std::copy_n(data(), m_size, heap);
		if (m_capacity != 0) {
			delete[] m_heap;
		}
		m_heap     = heap;
		m_capacity = static_cast<uint32_t>(grown);
	}

	uint32_t m_size     = 0;
	uint32_t m_capacity = 0; // 0: the inline storage is in use
	union {
		std::array<Use, InlineCapacity> m_inline;
		Use*                            m_heap;
	};
};

class Inst {
public:
	explicit Inst(ValueOpcode opcode, uint64_t flags = 0);
	~Inst();

	Inst(const Inst&)            = delete;
	Inst& operator=(const Inst&) = delete;
	Inst(Inst&&)                 = delete;
	Inst& operator=(Inst&&)      = delete;

	[[nodiscard]] ValueOpcode             GetOpcode() const;
	[[nodiscard]] Type                    GetType() const;
	[[nodiscard]] bool                    MayHaveSideEffects() const;
	[[nodiscard]] bool                    HasUses() const;
	[[nodiscard]] size_t                  UseCount() const;
	[[nodiscard]] size_t                  NumArgs() const;
	[[nodiscard]] size_t                  NumPhiBlocks() const;
	[[nodiscard]] Value                   Arg(size_t index) const;
	[[nodiscard]] Block*                  PhiBlock(size_t index) const;
	[[nodiscard]] Block*                  Parent() const;
	[[nodiscard]] const UseList&          Uses() const;
	// Runtime indices belong to the resource plan that owns this instruction.
	[[nodiscard]] uint32_t EvaluationIndex(uint32_t& count) const {
		if (evaluation_index == UINT32_MAX) {
			evaluation_index = count++;
		}
		return evaluation_index;
	}

	void SetParent(Block* block);
	void SetArg(size_t index, Value value);
	void AddPhiOperand(Block* predecessor, Value value);
	void ReplaceUsesWith(Value replacement, bool preserve = true);
	void Invalidate();

	template <typename T>
	requires(sizeof(T) <= sizeof(uint64_t) && std::is_trivially_copyable_v<T>)
	[[nodiscard]] T Flags() const {
		T result {};
		std::memcpy(&result, &flags, sizeof(result));
		return result;
	}

	template <typename T>
	requires(sizeof(T) <= sizeof(uint64_t) && std::is_trivially_copyable_v<T>)
	void SetFlags(T value) {
		flags = 0;
		std::memcpy(&flags, &value, sizeof(value));
	}

private:
	friend void EliminateDeadCode(const std::vector<Block*>& blocks);

	void AddUse(Inst* used, size_t operand);
	void RemoveUse(Inst* used, size_t operand);
	void ClearArgs();
	void StoreArg(size_t index, Value value); // the operand only, without use bookkeeping

	static constexpr uint8_t InlineArity = 4;
	static constexpr uint8_t PhiArity = UINT8_MAX;

	ValueOpcode         opcode;
	uint8_t             num_args;
	bool                live = false;
	mutable uint32_t    evaluation_index = UINT32_MAX;
	uint64_t            flags;
	Block*              parent = nullptr;
	union {
		std::array<Value, InlineArity> fixed_args {};
		std::vector<Value> large_args;
		std::vector<std::pair<Block*, Value>> phi_args;
	};
	UseList             uses;
};

// 120 bytes of operand storage plus the inline use list (UseList: two uses in place).
static_assert(sizeof(Inst) <= 136, "Inst operand storage unintentionally increased");

} // namespace Libs::Graphics::ShaderRecompiler::IR
