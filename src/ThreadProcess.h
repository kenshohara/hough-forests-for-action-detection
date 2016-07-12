#ifndef THREAD_PROCESS
#define THREAD_PROCESS

#include <functional>
#include <queue>

namespace nuisken {
namespace thread {

void threadProcess(std::queue<std::function<void()>>& tasks, int maxNumberOfThreads);
}
}

#endif