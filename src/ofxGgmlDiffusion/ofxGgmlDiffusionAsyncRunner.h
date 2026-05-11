#pragma once

#include "ofxGgmlDiffusionNativeBackend.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

enum class ofxGgmlDiffusionTaskState {
	Idle = 0,
	Running,
	Cancelling,
	Complete,
	Failed,
	Cancelled
};

class ofxGgmlDiffusionAsyncRunner {
public:
	ofxGgmlDiffusionAsyncRunner();
	~ofxGgmlDiffusionAsyncRunner();

	ofxGgmlDiffusionAsyncRunner(const ofxGgmlDiffusionAsyncRunner&) = delete;
	ofxGgmlDiffusionAsyncRunner& operator=(const ofxGgmlDiffusionAsyncRunner&) = delete;

	ofxGgmlDiffusionResult start(
		const ofxGgmlDiffusionContextSettings& settings,
		const ofxGgmlDiffusionRequest& request);
	void cancel();
	void wait();
	void close();

	bool isRunning() const;
	bool isDone() const;
	bool hasResult() const;
	ofxGgmlDiffusionTaskState getState() const;
	std::string getStatus() const;
	ofxGgmlDiffusionResult getResult() const;
	bool consumeResult(ofxGgmlDiffusionResult& result);

private:
	class Impl;
	std::unique_ptr<Impl> impl;
};

std::string ofxGgmlDiffusionGetTaskStateName(ofxGgmlDiffusionTaskState state);
