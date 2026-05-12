#pragma once

#include "ofxGgmlDiffusionImageGenerationBackend.h"

#include <memory>

class ofxGgmlDiffusionNativeBackend
	: public ofxGgmlDiffusionImageGenerationBackend {
public:
	ofxGgmlDiffusionNativeBackend();
	~ofxGgmlDiffusionNativeBackend();

	ofxGgmlDiffusionNativeBackend(ofxGgmlDiffusionNativeBackend&& other) noexcept;
	ofxGgmlDiffusionNativeBackend& operator=(ofxGgmlDiffusionNativeBackend&& other) noexcept;

	ofxGgmlDiffusionNativeBackend(const ofxGgmlDiffusionNativeBackend&) = delete;
	ofxGgmlDiffusionNativeBackend& operator=(const ofxGgmlDiffusionNativeBackend&) = delete;

	std::string getBackendName() const override;
	ofxGgmlDiffusionBackendFamily getBackendFamily() const override;
	bool isAvailable() const override;
	bool isLoaded() const override;
	ofxGgmlDiffusionContextSettings getSettings() const;

	ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings) override;
	ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request) override;
	void close();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend>
ofxGgmlMakeNativeDiffusionImageGenerationBackend();
