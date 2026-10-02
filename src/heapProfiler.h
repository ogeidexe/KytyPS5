#ifndef KYTY_HEAP_PROFILER_H_
#define KYTY_HEAP_PROFILER_H_

namespace Kyty {

// KYTY_HEAP_PROFILE=<path> (diagnostic, Windows): a sampling profile of the live C++ heap. Writes
// <path>.<n>.txt, the estimated live bytes per allocation call stack, largest first; a stack whose
// bytes keep growing from one report to the next is a leak or an unbounded cache. Does nothing
// when the variable is unset.
void HeapProfileWriteReport();

} // namespace Kyty

#endif /* KYTY_HEAP_PROFILER_H_ */
