#include "ThreadProcess.h"

#include <mutex>
#include <thread>

namespace nuisken {
namespace thread {
void threadProcess(std::queue<std::function<void()>>& tasks, int maxNumberOfThreads) {
    std::vector<std::thread> workers;
    std::mutex taskMutex;
    for (int i = 0; i < maxNumberOfThreads; ++i) {
        workers.push_back(std::thread([&taskMutex, &tasks]() {
            while (true) {
                std::function<void()> task;
                {  //タスク取得
                    std::lock_guard<std::mutex> taskLock(taskMutex);
                    if (tasks.empty()) {
                        break;
                    }

                    task = tasks.front();
                    tasks.pop();
                }

                task();
            }
        }));
    }

    for (auto& worker : workers) {
        worker.join();
    }
}
}
}