#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include <memory>

class ofxGgmlDiffusionNativeBackend {
public:
	ofxGgmlDiffusionNativeBackend();
	~ofxGgmlDiffusionNativeBackend();

	ofxGgmlDiffusionNativeBackend(ofxGgmlDiffusionNativeBackend&& other) noexcept;
	ofxGgmlDiffusionNativeBackend& operator=(ofxGgmlDiffusionNativeBackend&& other) noexcept;

	ofxGgmlDiffusionNativeBackend(const ofxGgmlDiffusionNativeBackend&) = delete;
	ofxGgmlDiffusionNativeBackend& operator=(const ofxGgmlDiffusionNativeBackend&) = delete;

	bool isAvailable() const;
	bool isLoaded() const;
	std::string getBackendName() const;
	ofxGgmlDiffusionContextSettings getSettings() const;

	ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings);
	ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request);
	void close();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
