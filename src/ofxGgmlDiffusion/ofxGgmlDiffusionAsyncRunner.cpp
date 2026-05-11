#include "ofxGgmlDiffusionAsyncRunner.h"

namespace {
	ofxGgmlDiffusionResult makeRunnerError(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	ofxGgmlDiffusionResult makeRunnerOk(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = true;
		result.text = message;
		return result;
	}
}

class ofxGgmlDiffusionAsyncRunner::Impl {
public:
	~Impl() {
		close();
	}

	ofxGgmlDiffusionResult start(
		const ofxGgmlDiffusionContextSettings& settings,
		const ofxGgmlDiffusionRequest& request) {
		waitFinishedWorker();
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (state == ofxGgmlDiffusionTaskState::Running ||
				state == ofxGgmlDiffusionTaskState::Cancelling) {
				return makeRunnerError("diffusion generation is already running");
			}
			cancelRequested.store(false);
			result = ofxGgmlDiffusionResult{};
			hasUnreadResult = false;
			state = ofxGgmlDiffusionTaskState::Running;
			status = "running";
		}

		worker = std::thread([this, settings, request]() {
			run(settings, request);
		});
		return makeRunnerOk("diffusion generation started");
	}

	void cancel() {
		cancelRequested.store(true);
		std::lock_guard<std::mutex> lock(mutex);
		if (state == ofxGgmlDiffusionTaskState::Running) {
			state = ofxGgmlDiffusionTaskState::Cancelling;
			status = "cancelling";
		}
	}

	void wait() {
		if (worker.joinable()) {
			worker.join();
		}
	}

	void close() {
		cancel();
		wait();
	}

	bool isRunning() const {
		std::lock_guard<std::mutex> lock(mutex);
		return state == ofxGgmlDiffusionTaskState::Running ||
			state == ofxGgmlDiffusionTaskState::Cancelling;
	}

	bool isDone() const {
		std::lock_guard<std::mutex> lock(mutex);
		return state == ofxGgmlDiffusionTaskState::Complete ||
			state == ofxGgmlDiffusionTaskState::Failed ||
			state == ofxGgmlDiffusionTaskState::Cancelled;
	}

	bool hasResult() const {
		std::lock_guard<std::mutex> lock(mutex);
		return hasUnreadResult;
	}

	ofxGgmlDiffusionTaskState getState() const {
		std::lock_guard<std::mutex> lock(mutex);
		return state;
	}

	std::string getStatus() const {
		std::lock_guard<std::mutex> lock(mutex);
		return status;
	}

	ofxGgmlDiffusionResult getResult() const {
		std::lock_guard<std::mutex> lock(mutex);
		return result;
	}

	bool consumeResult(ofxGgmlDiffusionResult& output) {
		std::lock_guard<std::mutex> lock(mutex);
		if (!hasUnreadResult) {
			return false;
		}
		output = result;
		hasUnreadResult = false;
		return true;
	}

private:
	void waitFinishedWorker() {
		if (!worker.joinable()) {
			return;
		}
		if (isRunning()) {
			return;
		}
		worker.join();
	}

	void run(
		const ofxGgmlDiffusionContextSettings& settings,
		const ofxGgmlDiffusionRequest& request) {
		ofxGgmlDiffusionNativeBackend backend;

		auto setupResult = backend.setup(settings);
		if (!setupResult) {
			finish(setupResult, ofxGgmlDiffusionTaskState::Failed, "setup failed");
			return;
		}

		if (cancelRequested.load()) {
			finish(makeRunnerError("diffusion generation cancelled"), ofxGgmlDiffusionTaskState::Cancelled, "cancelled");
			return;
		}

		auto generateResult = backend.generate(request);
		if (cancelRequested.load()) {
			finish(makeRunnerError("diffusion generation cancelled"), ofxGgmlDiffusionTaskState::Cancelled, "cancelled");
			return;
		}

		if (generateResult) {
			finish(generateResult, ofxGgmlDiffusionTaskState::Complete, "complete");
		} else {
			finish(generateResult, ofxGgmlDiffusionTaskState::Failed, "generation failed");
		}
	}

	void finish(
		const ofxGgmlDiffusionResult& nextResult,
		ofxGgmlDiffusionTaskState nextState,
		const std::string& nextStatus) {
		std::lock_guard<std::mutex> lock(mutex);
		result = nextResult;
		state = nextState;
		status = nextStatus;
		hasUnreadResult = true;
	}

	mutable std::mutex mutex;
	std::thread worker;
	std::atomic<bool> cancelRequested{false};
	ofxGgmlDiffusionTaskState state = ofxGgmlDiffusionTaskState::Idle;
	std::string status = "idle";
	ofxGgmlDiffusionResult result;
	bool hasUnreadResult = false;
};

ofxGgmlDiffusionAsyncRunner::ofxGgmlDiffusionAsyncRunner()
	: impl(std::make_unique<Impl>()) {
}

ofxGgmlDiffusionAsyncRunner::~ofxGgmlDiffusionAsyncRunner() = default;

ofxGgmlDiffusionResult ofxGgmlDiffusionAsyncRunner::start(
	const ofxGgmlDiffusionContextSettings& settings,
	const ofxGgmlDiffusionRequest& request) {
	return impl->start(settings, request);
}

void ofxGgmlDiffusionAsyncRunner::cancel() {
	impl->cancel();
}

void ofxGgmlDiffusionAsyncRunner::wait() {
	impl->wait();
}

void ofxGgmlDiffusionAsyncRunner::close() {
	impl->close();
}

bool ofxGgmlDiffusionAsyncRunner::isRunning() const {
	return impl->isRunning();
}

bool ofxGgmlDiffusionAsyncRunner::isDone() const {
	return impl->isDone();
}

bool ofxGgmlDiffusionAsyncRunner::hasResult() const {
	return impl->hasResult();
}

ofxGgmlDiffusionTaskState ofxGgmlDiffusionAsyncRunner::getState() const {
	return impl->getState();
}

std::string ofxGgmlDiffusionAsyncRunner::getStatus() const {
	return impl->getStatus();
}

ofxGgmlDiffusionResult ofxGgmlDiffusionAsyncRunner::getResult() const {
	return impl->getResult();
}

bool ofxGgmlDiffusionAsyncRunner::consumeResult(ofxGgmlDiffusionResult& result) {
	return impl->consumeResult(result);
}

std::string ofxGgmlDiffusionGetTaskStateName(ofxGgmlDiffusionTaskState state) {
	switch (state) {
	case ofxGgmlDiffusionTaskState::Idle: return "idle";
	case ofxGgmlDiffusionTaskState::Running: return "running";
	case ofxGgmlDiffusionTaskState::Cancelling: return "cancelling";
	case ofxGgmlDiffusionTaskState::Complete: return "complete";
	case ofxGgmlDiffusionTaskState::Failed: return "failed";
	case ofxGgmlDiffusionTaskState::Cancelled: return "cancelled";
	default: return "unknown";
	}
}
