#ifndef THREAD_PROCESS
#define THREAD_PROCESS

#include <queue>
#include <functional>

namespace nuisken {
namespace thread {

void threadProcess(std::queue<std::function<void()>>& tasks, int maxNumberOfThreads);
}
}

#endif